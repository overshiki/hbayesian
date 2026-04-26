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
  , runChain
  , runChainV2
  ) where

import           Data.Word           (Word64)
import           Data.Text           (Text)
import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import           HBayesian.MCMC.HMC
import           HBayesian.Chain

-- | Log-posterior for Bayesian logistic regression.
logisticRegLogPdf :: Tensor '[3] 'F32 -> Builder (Tensor '[] 'F32)
logisticRegLogPdf beta = do
  let mkConstX [a, b, c] = do
        ca <- tconstant @'[] @'F32 (realToFrac a)
        cb <- tconstant @'[] @'F32 (realToFrac b)
        cc <- tconstant @'[] @'F32 (realToFrac c)
        tpack3 ca cb cc
      mkConstX _ = error "mkConstX: expected exactly 3 elements"

  x0 <- mkConstX [1.0 :: Float, 0.5, -0.5]
  y0 <- tconstant @'[] @'F32 1.0
  lp0 <- logLikPoint x0 y0 beta

  x1 <- mkConstX [1.0 :: Float, 1.0, -1.0]
  y1 <- tconstant @'[] @'F32 1.0
  lp1 <- logLikPoint x1 y1 beta

  x2 <- mkConstX [1.0 :: Float, 1.5, -1.5]
  y2 <- tconstant @'[] @'F32 0.0
  lp2 <- logLikPoint x2 y2 beta

  x3 <- mkConstX [1.0 :: Float, 2.0, -2.0]
  y3 <- tconstant @'[] @'F32 0.0
  lp3 <- logLikPoint x3 y3 beta

  llh01 <- tadd lp0 lp1
  llh23 <- tadd lp2 lp3
  llh   <- tadd llh01 llh23

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

  grad0 <- gradPoint [1.0, 0.5, -0.5] 1.0 beta mkConstX
  grad1 <- gradPoint [1.0, 1.0, -1.0] 1.0 beta mkConstX
  grad2 <- gradPoint [1.0, 1.5, -1.5] 0.0 beta mkConstX
  grad3 <- gradPoint [1.0, 2.0, -2.0] 0.0 beta mkConstX

  g01 <- tadd grad0 grad1
  g23 <- tadd grad2 grad3
  gradLik <- tadd g01 g23

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

-- | Tier B: run a short chain on PJRT and return sampled beta vectors.
runChain :: IO [[Float]]
runChain = withPJRTCPU $ \api client -> do
    let config = HMCConfig { hmcStepSize = 0.1, hmcNumLeapfrogSteps = 2 }
        kernel = makeKernel config

    -- Compile the log-pdf module
    let ldMod = moduleFromBuilder @'[] @'F32 "main"
                  [ FuncArg "beta" (TensorType [3] F32) ] $ do
          beta <- arg @'[3] @'F32
          logisticRegLogPdf beta
    ldExe <- compileModule api client ldMod

    -- Compile the gradient module
    let gradMod = moduleFromBuilder @'[3] @'F32 "main"
                    [ FuncArg "beta" (TensorType [3] F32) ] $ do
          beta <- arg @'[3] @'F32
          logisticRegGrad beta
    gradExe <- compileModule api client gradMod

    -- Compile the HMC step module (single result: position)
    let stepMod = moduleFromBuilder @'[3] @'F32 "main"
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
          (state', _info) <- kernelStep kernel (Key key) (HMCState pos p ld g)
          return (hmcPosition state')
    stepExe <- compileModule api client stepMod

    let seed :: Word64 = 42
        beta0 = [0.0, 0.0, 0.0]

    -- Compute initial log-density and gradient
    betaBuf0 <- bufferFromF32 api client [3] beta0
    [ldBuf0] <- executeModule api ldExe [betaBuf0]
    [ld0] <- bufferToF32 api ldBuf0 1
    [gBuf0] <- executeModule api gradExe [betaBuf0]
    g0 <- bufferToF32 api gBuf0 3

    loop api client stepExe ldExe gradExe seed (0 :: Int) beta0 ld0 g0 (10 :: Int) []
  where
    loop _ _ _ _ _ _ _ _ _ _ 0 acc = return (reverse acc)
    loop api client stepExe ldExe gradExe seed step pos ld g n acc = do
        let key = [seed, fromIntegral step]
            zeroP = [0.0, 0.0, 0.0]
        keyBuf <- bufferFromUI64 api client [2] key
        posBuf <- bufferFromF32 api client [3] pos
        pBuf   <- bufferFromF32 api client [3] zeroP
        ldBuf  <- bufferFromF32 api client [] [ld]
        gBuf   <- bufferFromF32 api client [3] g
        [newPosBuf] <- executeModule api stepExe [keyBuf, posBuf, pBuf, ldBuf, gBuf]
        newPos <- bufferToF32 api newPosBuf 3
        -- Recompute log-density and gradient for the next step
        [newLdBuf] <- executeModule api ldExe [newPosBuf]
        [newLd] <- bufferToF32 api newLdBuf 1
        [newGBuf] <- executeModule api gradExe [newPosBuf]
        newG <- bufferToF32 api newGBuf 3
        loop api client stepExe ldExe gradExe seed (step + 1) newPos newLd newG (n - 1) (newPos : acc)

-- | v0.2: Run a chain using the 'Chain' combinators.
runChainV2 :: IO ([[Float]], [Diagnostic])
runChainV2 = do
    let config = HMCConfig { hmcStepSize = 0.1, hmcNumLeapfrogSteps = 2 }
        kernel = makeKernel config
        ck     = compileHMC kernel logisticRegLogPdf logisticRegGrad
    sampleChain ck [0.0, 0.0, 0.0] $ defaultChainConfig
        { ccNumIterations = 10
        , ccSeed = 42
        }
