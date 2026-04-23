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
  ) where

import           Data.Text           (Text)
import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import qualified HBayesian.HHLO.RNG as RNG
import           HBayesian.MCMC.HMC (HMCState(..))
import           HBayesian.MCMC.MALA
import           Common

-- | Precision matrix Lambda = Sigma^{-1} for
-- Sigma = [[1.0, 0.8], [0.8, 1.0]]
-- Lambda = [[2.7778, -2.2222], [-2.2222, 2.7778]]
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

  -- quadForm = d1*(lambda11*d1 + lambda12*d2) + d2*(lambda12*d1 + lambda22*d2)
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
