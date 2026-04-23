{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module HBayesian.MCMC.RandomWalk
  ( RWConfig (..)
  , randomWalk
  ) where

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import qualified HBayesian.HHLO.RNG as RNG

newtype RWConfig = RWConfig
  { rwScale :: Double
  }

randomWalk :: forall s d.
              (KnownShape s, KnownDType d)
           => (Tensor s d -> Builder (Tensor '[] d))
           -> RWConfig
           -> SimpleKernel s d
randomWalk logpdf config = Kernel { kernelInit = kernelInit, kernelStep = kernelStep }
  where
    scaleVal = rwScale config

    kernelInit _key pos = do
      ld <- logpdf pos
      return $ State pos ld

    kernelStep key state = do
      (key1, key2) <- RNG.splitKey key
      let pos = statePosition state
      let ld = stateLogDensity state

      noise <- RNG.rngNormalF32 key1 >>= convert @s @'F32 @d
      scaleT <- constant @s @d scaleVal
      scaledNoise <- tmul noise scaleT
      pos' <- tadd pos scaledNoise

      ld' <- logpdf pos'

      diff <- tsub ld' ld
      zero <- constant @'[] @d 0.0
      logAlpha <- tminimum diff zero

      u <- RNG.rngUniformF32 key2 >>= convert @'[] @'F32 @d
      logU <- tlog u
      accept <- tlessThan logU logAlpha
      acceptS <- tbroadcast @'[] @s [] accept

      newPos <- tselect acceptS pos' pos
      newLd  <- tselect accept ld' ld

      one <- constant @'[] @'I64 1
      infoAcceptProb <- texp logAlpha
      let info = Info infoAcceptProb accept one
      return (State newPos newLd, info)
