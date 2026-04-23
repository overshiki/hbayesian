{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module HBayesian.MCMC.MALA
  ( MALAConfig (..)
  , mala
  ) where

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import           HBayesian.HHLO.RNG
import           HBayesian.MCMC.HMC (HMCConfig (..), HMCState (..), hmc)

-- | Configuration for MALA.
newtype MALAConfig = MALAConfig
  { malaStepSize :: Double
  }

-- | MALA is HMC with a single leapfrog step.
mala :: forall s d.
        (KnownShape s, KnownDType d)
     => (Tensor s d -> Builder (Tensor '[] d))
     -> Gradient s d
     -> MALAConfig
     -> Kernel s d (HMCState s d) (Info s d)
mala logpdf grad config =
  hmc logpdf grad (HMCConfig (malaStepSize config) 1)
