{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 1: Bayesian Linear Regression with RandomWalk MH.
--
-- Model: y_i = alpha + beta * x_i + epsilon_i,  epsilon_i ~ N(0, 0.25)
-- Prior:  alpha ~ N(0, 1),  beta ~ N(0, 1)
module LinearRegressionRandomWalk
  ( dataset
  , linearRegLogPdf
  , makeKernel
  , renderStepMlir
  ) where

import           Data.Text           (Text)
import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import           HBayesian.MCMC.RandomWalk
import           Common

-- | Fixed synthetic dataset (n = 5).
-- True parameters: alpha = 1.0, beta = 1.5, sigma^2 = 0.25
dataset :: [(Float, Float)]
dataset =
  [ (0.0,  0.5)
  , (1.0,  2.0)
  , (2.0,  3.5)
  , (3.0,  5.0)
  , (4.0,  6.5)
  ]

-- | Log-posterior for Bayesian linear regression.
-- Parameters are packed as theta = [alpha, beta].
linearRegLogPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
linearRegLogPdf theta = do
  -- Unpack parameters
  alpha <- tslice1 @2 @'F32 theta 0
  beta  <- tslice1 @2 @'F32 theta 1

  -- Likelihood: sum over data points (unrolled for n=5)
  let likelihoodPoint (x, y) = do
        xT <- tconstant @'[] @'F32 (realToFrac x)
        yT <- tconstant @'[] @'F32 (realToFrac y)
        betaX <- tmul beta xT
        predVal <- tadd alpha betaX
        diff <- tsub yT predVal
        diffSq <- tmul diff diff
        -- -0.5 * diff^2 / 0.25 = -2.0 * diff^2
        negTwo <- tconstant @'[] @'F32 (-2.0)
        tmul negTwo diffSq

  llh0 <- likelihoodPoint (dataset !! 0)
  llh1 <- likelihoodPoint (dataset !! 1)
  llh2 <- likelihoodPoint (dataset !! 2)
  llh3 <- likelihoodPoint (dataset !! 3)
  llh4 <- likelihoodPoint (dataset !! 4)

  llh01 <- tadd llh0 llh1
  llh23 <- tadd llh2 llh3
  llh0123 <- tadd llh01 llh23
  llh <- tadd llh0123 llh4

  -- Prior: log N(alpha|0,1) + log N(beta|0,1)
  alphaSq <- tmul alpha alpha
  betaSq  <- tmul beta beta
  negHalf <- tconstant @'[] @'F32 (-0.5)
  priorAlpha <- tmul negHalf alphaSq
  priorBeta  <- tmul negHalf betaSq

  tadd llh =<< tadd priorAlpha priorBeta

-- | Factory: build a RandomWalk kernel for this model.
makeKernel :: RWConfig -> SimpleKernel '[2] 'F32
makeKernel config = randomWalk linearRegLogPdf config

-- | Tier A: render one kernel step to MLIR text.
renderStepMlir :: Text
renderStepMlir =
  renderKernelStep @'[2] @'F32
    [ FuncArg "key" (TensorType [2] UI64)
    , FuncArg "pos" (TensorType [2] F32)
    , FuncArg "ld"  (TensorType [] F32)
    ] $ do
      key <- arg @'[2] @'UI64
      pos <- arg @'[2] @'F32
      ld  <- arg @'[] @'F32
      (state', _info) <- kernelStep (makeKernel (RWConfig 0.1)) (Key key) (State pos ld)
      return (statePosition state')
