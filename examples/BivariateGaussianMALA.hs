{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 4: Bivariate Gaussian Target with MALA.
--
-- A simple 2D correlated Gaussian where MALA's single leapfrog step
-- is competitive.  The gradient is trivial and closed-form.
module BivariateGaussianMALA
  ( bivariateLogPdf
  , bivariateGrad
  , makeKernel
  , renderStepMlir
  , runChain
  ) where

import           Data.Word           (Word64)
import           Data.Text           (Text)
import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import           HBayesian.HHLO.PJRT
import           HBayesian.MCMC.HMC (HMCState(..))
import           HBayesian.MCMC.MALA
import           Common

-- | Precision matrix Lambda = Sigma^{-1} for
-- Sigma = [[1.0, 0.8], [0.8, 1.0]]
lambda11, lambda12, lambda22 :: Float
lambda11 = 2.7778
lambda12 = -2.2222
lambda22 = 2.7778

mu1, mu2 :: Float
mu1 = 1.0
mu2 = 2.0

-- | Log-density of the bivariate Gaussian.
bivariateLogPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
bivariateLogPdf theta = do
  mu <- buildMu
  diff <- tsub theta mu

  d1 <- tslice1 @2 @'F32 diff 0
  d2 <- tslice1 @2 @'F32 diff 1

  l11 <- tconstant @'[] @'F32 (realToFrac lambda11)
  l12 <- tconstant @'[] @'F32 (realToFrac lambda12)
  l22 <- tconstant @'[] @'F32 (realToFrac lambda22)

  term1a <- tmul l11 d1
  term1b <- tmul l12 d2
  term1  <- tadd term1a term1b
  quad1  <- tmul d1 term1

  term2a <- tmul l12 d1
  term2b <- tmul l22 d2
  term2  <- tadd term2a term2b
  quad2  <- tmul d2 term2

  quadForm <- tadd quad1 quad2
  negHalf <- tconstant @'[] @'F32 (-0.5)
  tmul negHalf quadForm

-- | Gradient of the bivariate Gaussian log-density.
-- grad = -Lambda * (theta - mu)
bivariateGrad :: Gradient '[2] 'F32
bivariateGrad theta = do
  mu <- buildMu
  diff <- tsub theta mu

  d1 <- tslice1 @2 @'F32 diff 0
  d2 <- tslice1 @2 @'F32 diff 1

  l11 <- tconstant @'[] @'F32 (realToFrac lambda11)
  l12 <- tconstant @'[] @'F32 (realToFrac lambda12)
  l22 <- tconstant @'[] @'F32 (realToFrac lambda22)

  g1a <- tmul l11 d1
  g1b <- tmul l12 d2
  g1Inner <- tadd g1a g1b
  g1 <- tnegate g1Inner

  g2a <- tmul l12 d1
  g2b <- tmul l22 d2
  g2Inner <- tadd g2a g2b
  g2 <- tnegate g2Inner

  tpack2 g1 g2

-- | Helper: build the mean vector as a constant tensor.
buildMu :: Builder (Tensor '[2] 'F32)
buildMu = do
  m1 <- tconstant @'[] @'F32 (realToFrac mu1)
  m2 <- tconstant @'[] @'F32 (realToFrac mu2)
  tpack2 m1 m2

-- | Factory: build a MALA kernel for this target.
makeKernel :: MALAConfig -> Kernel '[2] 'F32 (HMCState '[2] 'F32) (Info '[2] 'F32)
makeKernel config = mala bivariateLogPdf bivariateGrad config

-- | Tier A: render one kernel step to MLIR text.
renderStepMlir :: Text
renderStepMlir =
  renderKernelStep @'[2] @'F32
    [ FuncArg "key" (TensorType [2] UI64)
    , FuncArg "pos" (TensorType [2] F32)
    , FuncArg "p"   (TensorType [2] F32)
    , FuncArg "ld"  (TensorType [] F32)
    , FuncArg "g"   (TensorType [2] F32)
    ] $ do
      key <- arg @'[2] @'UI64
      pos <- arg @'[2] @'F32
      p   <- arg @'[2] @'F32
      ld  <- arg @'[] @'F32
      g   <- arg @'[2] @'F32
      let config = MALAConfig { malaStepSize = 0.1 }
      (state', _info) <- kernelStep (makeKernel config) (Key key) (HMCState pos p ld g)
      return (hmcPosition state')

-- | Tier B: run a short chain on PJRT and return sampled positions.
runChain :: IO [[Float]]
runChain = withPJRTCPU $ \api client -> do
    let config = MALAConfig { malaStepSize = 0.1 }
        kernel = makeKernel config

    -- Compile the log-pdf module
    let ldMod = moduleFromBuilder @'[] @'F32 "main"
                  [ FuncArg "theta" (TensorType [2] F32) ] $ do
          theta <- arg @'[2] @'F32
          bivariateLogPdf theta
    ldExe <- compileModule api client ldMod

    -- Compile the gradient module
    let gradMod = moduleFromBuilder @'[2] @'F32 "main"
                    [ FuncArg "theta" (TensorType [2] F32) ] $ do
          theta <- arg @'[2] @'F32
          bivariateGrad theta
    gradExe <- compileModule api client gradMod

    -- Compile the MALA step module (single result: position)
    let stepMod = moduleFromBuilder @'[2] @'F32 "main"
                    [ FuncArg "key" (TensorType [2] UI64)
                    , FuncArg "pos" (TensorType [2] F32)
                    , FuncArg "p"   (TensorType [2] F32)
                    , FuncArg "ld"  (TensorType [] F32)
                    , FuncArg "g"   (TensorType [2] F32)
                    ] $ do
          key <- arg @'[2] @'UI64
          pos <- arg @'[2] @'F32
          p   <- arg @'[2] @'F32
          ld  <- arg @'[] @'F32
          g   <- arg @'[2] @'F32
          (state', _info) <- kernelStep kernel (Key key) (HMCState pos p ld g)
          return (hmcPosition state')
    stepExe <- compileModule api client stepMod

    let seed :: Word64 = 42
        theta0 = [0.0, 0.0]

    -- Compute initial log-density and gradient
    thetaBuf0 <- bufferFromF32 api client [2] theta0
    [ldBuf0] <- executeModule api ldExe [thetaBuf0]
    [ld0] <- bufferToF32 api ldBuf0 1
    [gBuf0] <- executeModule api gradExe [thetaBuf0]
    g0 <- bufferToF32 api gBuf0 2

    loop api client stepExe ldExe gradExe seed (0 :: Int) theta0 ld0 g0 (10 :: Int) []
  where
    loop _ _ _ _ _ _ _ _ _ _ 0 acc = return (reverse acc)
    loop api client stepExe ldExe gradExe seed step pos ld g n acc = do
        let key = [seed, fromIntegral step]
            zeroP = [0.0, 0.0]
        keyBuf <- bufferFromUI64 api client [2] key
        posBuf <- bufferFromF32 api client [2] pos
        pBuf   <- bufferFromF32 api client [2] zeroP
        ldBuf  <- bufferFromF32 api client [] [ld]
        gBuf   <- bufferFromF32 api client [2] g
        [newPosBuf] <- executeModule api stepExe [keyBuf, posBuf, pBuf, ldBuf, gBuf]
        newPos <- bufferToF32 api newPosBuf 2
        -- Recompute log-density and gradient for the next step
        [newLdBuf] <- executeModule api ldExe [newPosBuf]
        [newLd] <- bufferToF32 api newLdBuf 1
        [newGBuf] <- executeModule api gradExe [newPosBuf]
        newG <- bufferToF32 api newGBuf 2
        loop api client stepExe ldExe gradExe seed (step + 1) newPos newLd newG (n - 1) (newPos : acc)
