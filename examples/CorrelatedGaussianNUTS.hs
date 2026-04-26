{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example: NUTS on a 5-D correlated Gaussian with AR(1) covariance.
--
-- This example demonstrates the No-U-Turn Sampler on the same target
-- distribution as CorrelatedGaussianHMC, allowing direct comparison.
module CorrelatedGaussianNUTS
  ( runChainV2
  ) where

import           HHLO.Core.Types
import           HBayesian.Core
import           HBayesian.MCMC.NUTS
import           HBayesian.Chain

-- Re-use the model from CorrelatedGaussianHMC
import           CorrelatedGaussianHMC
  ( targetDim
  , gaussianLogPdf
  , gaussianGrad
  )

makeKernel :: NUTSConfig -> Kernel '[5] 'F32 (NUTSState '[5] 'F32) (Info '[5] 'F32)
makeKernel config = nuts gaussianLogPdf gaussianGrad config

runChainV2 :: IO ([[Float]], [Diagnostic])
runChainV2 = do
    let config = NUTSConfig { nutsStepSize = 0.1, nutsMaxDepth = 5, nutsDeltaMax = 1000.0 }
        kernel = makeKernel config
        ck     = compileNUTS kernel gaussianLogPdf gaussianGrad
    sampleChain ck (replicate targetDim 0.0) $
        burnIn 200 $ thin 2 $ defaultChainConfig
            { ccNumIterations = 1000
            , ccSeed = 42
            }
