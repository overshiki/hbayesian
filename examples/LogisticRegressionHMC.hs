{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 3: Bayesian Logistic Regression with HMC.
--
-- Binary classification with D=3 features and n=4 observations.
-- The user provides the gradient explicitly, demonstrating the
-- manual-gradient path before Phase 5 auto-diff.
module LogisticRegressionHMC
  ( logisticRegLogPdf
  , logisticRegGrad
  , makeKernel
  , renderStepMlir
  ) where

import           Data.Text           (Text)
import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import           HBayesian.MCMC.HMC
import           Common

-- | Log-posterior for Bayesian logistic regression.
logisticRegLogPdf :: Tensor '[3] 'F32 -> Builder (Tensor '[] 'F32)
logisticRegLogPdf beta = do
  let mkConstX [a, b, c] = do
        ca <- tconstant @'[] @'F32 (realToFrac a)
        cb <- tconstant @'[] @'F32 (realToFrac b)
        cc <- tconstant @'[] @'F32 (realToFrac c)
        tpack3 ca cb cc
      mkConstX _ = error "mkConstX: expected exactly 3 elements"

  -- Data point 1
  x0 <- mkConstX [1.0 :: Float, 0.5, -0.5]
  y0 <- tconstant @'[] @'F32 1.0
  lp0 <- logLikPoint x0 y0 beta

  -- Data point 2
  x1 <- mkConstX [1.0 :: Float, 1.0, -1.0]
  y1 <- tconstant @'[] @'F32 1.0
  lp1 <- logLikPoint x1 y1 beta

  -- Data point 3
  x2 <- mkConstX [1.0 :: Float, 1.5, -1.5]
  y2 <- tconstant @'[] @'F32 0.0
  lp2 <- logLikPoint x2 y2 beta

  -- Data point 4
  x3 <- mkConstX [1.0 :: Float, 2.0, -2.0]
  y3 <- tconstant @'[] @'F32 0.0
  lp3 <- logLikPoint x3 y3 beta

  llh01 <- tadd lp0 lp1
  llh23 <- tadd lp2 lp3
  llh   <- tadd llh01 llh23

  -- Prior: -0.5 * beta^T beta
  betaSq <- tmul beta beta
  betaSumSq <- tsumAll betaSq
  negHalf <- tconstant @'[] @'F32 (-0.5)
  prior <- tmul negHalf betaSumSq

  tadd llh prior

-- | Single data-point log-likelihood contribution.
logLikPoint :: Tensor '[3] 'F32 -> Tensor '[] 'F32 -> Tensor '[3] 'F32 -> Builder (Tensor '[] 'F32)
logLikPoint x_i y_i beta = do
  dotProd <- tsumAll =<< tmul x_i beta
  sig <- tsigmoid dotProd
  one <- tconstant @'[] @'F32 1.0
  logSig   <- tlog sig
  logOneMinusSig <- tlog =<< tsub one sig
  term1 <- tmul y_i logSig
  oneMinusY <- tsub one y_i
  term2 <- tmul oneMinusY logOneMinusSig
  tadd term1 term2

-- | User-provided gradient of the log-posterior.
logisticRegGrad :: Gradient '[3] 'F32
logisticRegGrad beta = do
  let mkConstX [a, b, c] = do
        ca <- tconstant @'[] @'F32 (realToFrac a)
        cb <- tconstant @'[] @'F32 (realToFrac b)
        cc <- tconstant @'[] @'F32 (realToFrac c)
        tpack3 ca cb cc
      mkConstX _ = error "mkConstX: expected exactly 3 elements"

  -- Gradient from each data point: (y_i - sigmoid(x_i^T beta)) * x_i
  grad0 <- gradPoint [1.0, 0.5, -0.5] 1.0 beta mkConstX
  grad1 <- gradPoint [1.0, 1.0, -1.0] 1.0 beta mkConstX
  grad2 <- gradPoint [1.0, 1.5, -1.5] 0.0 beta mkConstX
  grad3 <- gradPoint [1.0, 2.0, -2.0] 0.0 beta mkConstX

  g01 <- tadd grad0 grad1
  g23 <- tadd grad2 grad3
  gradLik <- tadd g01 g23

  -- Prior gradient: -beta
  negOne <- tconstant @'[3] @'F32 (-1.0)
  gradPrior <- tmul negOne beta

  tadd gradLik gradPrior

-- | Gradient contribution from a single data point.
gradPoint :: [Float] -> Float -> Tensor '[3] 'F32
          -> ([Float] -> Builder (Tensor '[3] 'F32))
          -> Builder (Tensor '[3] 'F32)
gradPoint xs yi beta mkConstX = do
  x_i <- mkConstX xs
  dotProd <- tsumAll =<< tmul x_i beta
  sig <- tsigmoid dotProd
  yT <- tconstant @'[] @'F32 (realToFrac yi)
  residual <- tsub yT sig
  -- Broadcast residual to [3] and multiply element-wise with x_i
  residBC <- tbroadcast @'[] @'[3] [] residual
  tmul x_i residBC

-- | Factory: build an HMC kernel for this model.
makeKernel :: HMCConfig -> Kernel '[3] 'F32 (HMCState '[3] 'F32) (Info '[3] 'F32)
makeKernel config = hmc logisticRegLogPdf logisticRegGrad config

-- | Tier A: render one kernel step to MLIR text.
renderStepMlir :: Text
renderStepMlir =
  renderKernelStep @'[3] @'F32
    [ FuncArg "key" (TensorType [2] UI64)
    , FuncArg "pos" (TensorType [3] F32)
    , FuncArg "p"   (TensorType [3] F32)
    , FuncArg "ld"  (TensorType [] F32)
    , FuncArg "g"   (TensorType [3] F32)
    ] $ do
      key <- arg @'[2] @'UI64
      pos <- arg @'[3] @'F32
      p   <- arg @'[3] @'F32
      ld  <- arg @'[] @'F32
      g   <- arg @'[3] @'F32
      let config = HMCConfig { hmcStepSize = 0.1, hmcNumLeapfrogSteps = 2 }
      (state', _info) <- kernelStep (makeKernel config) (Key key) (HMCState pos p ld g)
      return (hmcPosition state')
