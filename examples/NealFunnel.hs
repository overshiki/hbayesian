{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Neal's Funnel: the canonical distribution where NUTS dramatically
-- outperforms fixed-step HMC.
--
-- Model (Stan formulation):
--   y ~ N(0, 3^2)
--   x_i ~ N(0, exp(y/2)^2)  for i = 1..n
--
-- The funnel geometry changes dramatically: at small y, x's are tightly
-- concentrated; at large y, x's are very diffuse. No fixed trajectory
-- length works well across both regimes.
module NealFunnel
  ( funnelDim
  , funnelLogPdf
  , funnelGrad
  , makeHMCKernel
  , makeNUTSKernel
  , runHMC
  , runNUTS
  ) where

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops hiding (map, maximum, sort, sqrt)
import           HBayesian.MCMC.HMC
import           HBayesian.MCMC.NUTS
import           HBayesian.Chain

-- | Total dimensionality: 1 scale parameter + 9 latent variables.
funnelDim :: Int
funnelDim = 10

-----------------------------------------------------------------------------
-- Log-density
-----------------------------------------------------------------------------

-- | log p(y, x) = -y^2/18 - sum_i[x_i^2 * exp(-y) / 2] - 9*y/2 + const
funnelLogPdf :: Tensor '[10] 'F32 -> Builder (Tensor '[] 'F32)
funnelLogPdf theta = do
    y <- tslice1 @10 @'F32 theta 0

    -- y term: -y^2 / 18
    ySq <- tmul y y
    yTermCoeff <- constant @'[] @'F32 (-1.0 / 18.0)
    yTerm <- tmul ySq yTermCoeff

    -- exp(-y)
    negY <- tnegate y
    expNegY <- texp negY

    -- x terms: -sum_i[x_i^2] * exp(-y) / 2
    -- Compute sum of squares of all elements, then subtract y^2
    thetaSq <- tmul theta theta
    totalSq <- tsumAll thetaSq
    xSqSum <- tsub totalSq ySq
    xCoeffVal <- constant @'[] @'F32 (-0.5)
    xCoeff <- tmul expNegY xCoeffVal
    xTerm <- tmul xSqSum xCoeff

    -- log scale term: -9*y/2
    logScaleCoeff <- constant @'[] @'F32 (-9.0 / 2.0)
    logScale <- tmul y logScaleCoeff

    s1 <- tadd yTerm xTerm
    tadd s1 logScale

-----------------------------------------------------------------------------
-- Gradient
-----------------------------------------------------------------------------

-- | d/dy   = -y/9 - 9/2 + sum_i[x_i^2 * exp(-y) / 2]
--   d/dx_i = -x_i * exp(-y)
funnelGrad :: Gradient '[10] 'F32
funnelGrad theta = do
    y <- tslice1 @10 @'F32 theta 0

    negY <- tnegate y
    expNegY <- texp negY

    -- Common factor: -exp(-y), broadcast to shape [10]
    negExpNegY <- tnegate expNegY
    negExpNegYBC <- tbroadcast @'[] @'[10] [] negExpNegY

    -- gxAll[i] = -theta[i] * exp(-y) for all i
    gxAll <- tmul theta negExpNegYBC

    -- gy = -y/9 - 9/2 + sum_i[x_i^2 * exp(-y) / 2]
    thetaSq <- tmul theta theta
    totalSq <- tsumAll thetaSq
    ySq <- tmul y y
    xSqSum <- tsub totalSq ySq
    halfExpNegYVal <- constant @'[] @'F32 (0.5)
    halfExpNegY <- tmul expNegY halfExpNegYVal
    xSqTerm <- tmul xSqSum halfExpNegY
    yCoeff1Val <- constant @'[] @'F32 (-1.0 / 9.0)
    yCoeff1 <- tmul y yCoeff1Val
    yCoeff2 <- constant @'[] @'F32 (-9.0 / 2.0)
    yTemp <- tadd yCoeff1 yCoeff2
    gy <- tadd yTemp xSqTerm

    -- Extract all elements of gxAll
    g1 <- tslice1 @10 @'F32 gxAll 1
    g2 <- tslice1 @10 @'F32 gxAll 2
    g3 <- tslice1 @10 @'F32 gxAll 3
    g4 <- tslice1 @10 @'F32 gxAll 4
    g5 <- tslice1 @10 @'F32 gxAll 5
    g6 <- tslice1 @10 @'F32 gxAll 6
    g7 <- tslice1 @10 @'F32 gxAll 7
    g8 <- tslice1 @10 @'F32 gxAll 8
    g9 <- tslice1 @10 @'F32 gxAll 9

    -- Reshape and concatenate with corrected first element
    g0r <- treshape @'[] @'[1] gy
    g1r <- treshape @'[] @'[1] g1
    g2r <- treshape @'[] @'[1] g2
    g3r <- treshape @'[] @'[1] g3
    g4r <- treshape @'[] @'[1] g4
    g5r <- treshape @'[] @'[1] g5
    g6r <- treshape @'[] @'[1] g6
    g7r <- treshape @'[] @'[1] g7
    g8r <- treshape @'[] @'[1] g8
    g9r <- treshape @'[] @'[1] g9

    concatenate @'[1] @'[10] @'F32 0
        [g0r, g1r, g2r, g3r, g4r, g5r, g6r, g7r, g8r, g9r]

-----------------------------------------------------------------------------
-- Kernels
-----------------------------------------------------------------------------

makeHMCKernel :: Int -> Kernel '[10] 'F32 (HMCState '[10] 'F32) (Info '[10] 'F32)
makeHMCKernel leapfrogSteps = hmc funnelLogPdf funnelGrad config
  where
    config = HMCConfig { hmcStepSize = 0.05, hmcNumLeapfrogSteps = leapfrogSteps }

makeNUTSKernel :: Kernel '[10] 'F32 (NUTSState '[10] 'F32) (Info '[10] 'F32)
makeNUTSKernel = nuts funnelLogPdf funnelGrad config
  where
    config = NUTSConfig { nutsStepSize = 0.05, nutsMaxDepth = 10, nutsDeltaMax = 1000.0 }

-----------------------------------------------------------------------------
-- Runners
-----------------------------------------------------------------------------

runHMC :: Int -> IO ([[Float]], [Diagnostic])
runHMC leapfrogSteps = do
    let kernel = makeHMCKernel leapfrogSteps
        ck     = compileHMC kernel funnelLogPdf funnelGrad
    sampleChain ck (replicate funnelDim 0.0) $
        burnIn 500 $ thin 2 $ defaultChainConfig
            { ccNumIterations = 1000
            , ccSeed = 42
            }

runNUTS :: IO ([[Float]], [Diagnostic])
runNUTS = do
    let ck = compileNUTS makeNUTSKernel funnelLogPdf funnelGrad
    sampleChain ck (replicate funnelDim 0.0) $
        burnIn 500 $ thin 2 $ defaultChainConfig
            { ccNumIterations = 1000
            , ccSeed = 42
            }
