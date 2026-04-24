{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module HBayesian.MCMC.HMC
  ( HMCConfig (..)
  , HMCState (..)
  , hmc
  ) where

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import qualified HBayesian.HHLO.RNG as RNG

-- | Configuration for Hamiltonian Monte Carlo.
data HMCConfig = HMCConfig
  { hmcStepSize         :: Double
  , hmcNumLeapfrogSteps :: Int
  }

-- | HMC-specific state.
data HMCState (s :: Shape) (d :: DType) = HMCState
  { hmcPosition  :: !(Tensor s d)
  , hmcMomentum  :: !(Tensor s d)
  , hmcLogDens   :: !(Tensor '[] d)
  , hmcGradient  :: !(Tensor s d)
  }

hmc :: forall s d.
       (KnownShape s, KnownDType d)
    => (Tensor s d -> Builder (Tensor '[] d))
    -> Gradient s d
    -> HMCConfig
    -> Kernel s d (HMCState s d) (Info s d)
hmc logpdf grad config = Kernel { kernelInit = kernelInit, kernelStep = kernelStep }
  where
    epsVal = hmcStepSize config
    nSteps = hmcNumLeapfrogSteps config

    kernelInit _key pos = do
      ld <- logpdf pos
      g <- grad pos
      zeroM <- tconstant 0.0
      return $ HMCState pos zeroM ld g

    kernelStep key state = do
      (key1, key2) <- RNG.splitKey key
      let pos0 = hmcPosition state
      let g0   = hmcGradient state
      let ld0  = hmcLogDens state

      p0 <- RNG.rngNormalF32 key1 >>= convert @s @'F32 @d

      currentK <- do
        pSq <- tmul p0 p0
        pSum <- tsumAll pSq
        half <- constant @'[] @d 0.5
        tmul half pSum
      currentH <- tsub currentK ld0

      (pos', p', g', ld') <- leapfrog pos0 p0 g0

      proposedK <- do
        pSq <- tmul p' p'
        pSum <- tsumAll pSq
        half <- constant @'[] @d 0.5
        tmul half pSum
      proposedH <- tsub proposedK ld'

      logAccept <- tsub currentH proposedH
      zero <- constant @'[] @d 0.0
      logAlpha <- tminimum logAccept zero

      u <- RNG.rngUniformF32 key2 >>= convert @'[] @'F32 @d
      logU <- tlog u
      accept <- tlessThan logU logAlpha
      acceptS <- tbroadcast @'[] @s [] accept

      newPos <- tselect acceptS pos' pos0
      newP   <- tselect acceptS p' p0
      newG   <- tselect acceptS g' g0
      newLd  <- tselect accept ld' ld0

      infoAcceptProb <- texp logAlpha
      infoNumSteps <- constant @'[] @'I64 (fromIntegral nSteps)
      let info = Info infoAcceptProb accept infoNumSteps
      return (HMCState newPos newP newLd newG, info)

    leapfrog :: Tensor s d -> Tensor s d -> Tensor s d
             -> Builder (Tensor s d, Tensor s d, Tensor s d, Tensor '[] d)
    leapfrog pos0 p0 g0 = go nSteps pos0 p0 g0
      where
        go 0 pos p g = do
          ld <- logpdf pos
          return (pos, p, g, ld)
        go k pos p g = do
          halfEps <- constant @s @d (epsVal / 2.0)
          gScaled <- tmul g halfEps
          pHalf <- tadd p gScaled

          epsT <- constant @s @d epsVal
          pScaled <- tmul pHalf epsT
          pos' <- tadd pos pScaled

          g' <- grad pos'

          gScaled' <- tmul g' halfEps
          p' <- tadd pHalf gScaled'

          go (k-1) pos' p' g'
