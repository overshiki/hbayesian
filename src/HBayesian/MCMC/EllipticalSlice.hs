{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module HBayesian.MCMC.EllipticalSlice
  ( ellipticalSlice
  ) where

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import qualified HBayesian.HHLO.RNG as RNG

ellipticalSlice :: forall s d.
                   (KnownShape s, KnownDType d)
                => (Tensor s d -> Builder (Tensor '[] d))
                -> SimpleKernel s d
ellipticalSlice logpdf = Kernel { kernelInit = kernelInit, kernelStep = kernelStep }
  where
    kernelInit _key pos = do
      ld <- logpdf pos
      return $ State pos ld

    kernelStep key state = do
      (key1, key2) <- RNG.splitKey key
      let pos = statePosition state
      let ld = stateLogDensity state

      nu <- RNG.rngNormalF32 key1 >>= convert @s @'F32 @d

      thetaRaw <- RNG.rngUniformF32 key2 >>= convert @'[] @'F32 @d
      twoPi <- constant @'[] @d (2.0 * pi)
      theta <- tmul thetaRaw twoPi

      c <- tcos theta
      sTheta <- tsin theta
      cBC <- tbroadcast @'[] @s [] c
      sBC <- tbroadcast @'[] @s [] sTheta
      pc <- tmul pos cBC
      ns <- tmul nu sBC
      pos' <- tadd pc ns

      ld' <- logpdf pos'

      zero <- constant @'[] @d 0.0
      diff <- tsub ld' ld
      accept <- tlessThan zero diff
      acceptS <- tbroadcast @'[] @s [] accept

      newPos <- tselect acceptS pos' pos
      newLd  <- tselect accept ld' ld

      one <- constant @'[] @'I64 1
      infoAcceptProb <- texp diff
      let info = Info infoAcceptProb accept one
      return (State newPos newLd, info)
