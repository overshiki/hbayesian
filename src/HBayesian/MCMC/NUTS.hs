{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module HBayesian.MCMC.NUTS
  ( NUTSConfig (..)
  , NUTSState (..)
  , nuts
  ) where

import           Data.Int            (Int64)
import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.Builder
import           Prelude hiding (negate, minimum, maximum, sqrt, sin, cos, tan, floor, ceiling)

import           HBayesian.Core
import           HBayesian.HHLO.Ops
import qualified HBayesian.HHLO.RNG as RNG

-----------------------------------------------------------------------------
-- Configuration and state
-----------------------------------------------------------------------------

-- | Configuration for the No-U-Turn Sampler.
data NUTSConfig = NUTSConfig
  { nutsStepSize :: Double    -- ^ Leapfrog step size epsilon
  , nutsMaxDepth :: Int       -- ^ Maximum tree depth (default 10)
  , nutsDeltaMax :: Double    -- ^ Divergence threshold (default 1000.0)
  }

-- | NUTS-specific algorithm state.
data NUTSState (s :: Shape) (d :: DType) = NUTSState
  { nutsPosition :: !(Tensor s d)
  , nutsMomentum :: !(Tensor s d)
  , nutsLogDens  :: !(Tensor '[] d)
  , nutsGradient :: !(Tensor s d)
  }

-----------------------------------------------------------------------------
-- Main kernel
-----------------------------------------------------------------------------

nuts :: forall s d.
        (KnownShape s, KnownDType d)
     => (Tensor s d -> Builder (Tensor '[] d))   -- ^ log-posterior
     -> Gradient s d                              -- ^ gradient
     -> NUTSConfig
     -> Kernel s d (NUTSState s d) (Info s d)
nuts logpdf grad config = Kernel { kernelInit = kernelInit, kernelStep = kernelStep }
  where
    epsVal    = nutsStepSize config
    maxDepth  = nutsMaxDepth config
    deltaMaxF = nutsDeltaMax config

    kernelInit _key pos = do
      ld <- logpdf pos
      g  <- grad pos
      zeroM <- tconstant 0.0
      return $ NUTSState pos zeroM ld g

    kernelStep key state = do
      (key1, keyMom) <- RNG.splitKey key
      let pos0 = nutsPosition state

      -- 1. Resample momentum
      p0 <- RNG.rngNormalF32 keyMom >>= convert @s @'F32 @d

      -- 2. Compute initial Hamiltonian and slice variable
      h0 <- hamiltonian logpdf pos0 p0
      (key2, keySlice) <- RNG.splitKey key1
      upper <- texp =<< tnegate h0
      uRaw  <- RNG.rngUniformF32 keySlice >>= convert @'[] @'F32 @d
      u     <- tmul uRaw upper

      -- 3. Initialize tree state
      zeroI64 <- constant @'[] @'I64 0
      oneI64  <- constant @'[] @'I64 1
      packed0 <- tpack3 oneI64 oneI64 zeroI64   -- n=1, s=1, j=0

      -- Unwrap Key for whileLoop7
      let Key keyTensor = key2

      -- 4. Outer tree-doubling loop (whileLoop7)
      --    Carries: q-, p-, q+, p+, q', packed(n,s,j), key
      Tuple7 qmFin pmFin qpFin ppFin qcFin packedFin _keyFin <-
        whileLoop7 pos0 p0 pos0 p0 pos0 packed0 keyTensor
          (\_ _ _ _ _ packed _ -> do
              sFlg <- tslice1 packed 1
              jDep <- tslice1 packed 2
              -- s == 1 AND j < maxDepth
              sOk   <- equal sFlg oneI64
              jDepF <- convert @'[] @'I64 @'F32 jDep
              maxDepthT <- constant @'[] @'F32 (fromIntegral maxDepth)
              jOk   <- tlessThan jDepF maxDepthT
              tand sOk jOk)
          (\qm pm qp pp qc packed k -> do
              -- Unpack
              nAcc <- tslice1 packed 0
              sFlg <- tslice1 packed 1
              jDep <- tslice1 packed 2

              -- Sample direction: forward with prob 0.5
              (kDirKey, kRestKey) <- RNG.splitKey (Key k)
              vRaw <- RNG.rngUniformF32 kDirKey >>= convert @'[] @'F32 @d
              half <- constant @'[] @d 0.5
              vForward <- tlessThan vRaw half
              oneD    <- constant @'[] @d 1.0
              negOneD <- constant @'[] @d (-1.0)
              vScalar <- tselect vForward oneD negOneD
              epsT    <- constant @'[] @d epsVal
              epsV    <- tmul epsT vScalar

              -- Target steps: 2^j
              jDepF    <- convert @'[] @'I64 @'F32 jDep
              twoF     <- constant @'[] @'F32 2.0
              targetF  <- tpow twoF jDepF
              targetSteps <- convert @'[] @'F32 @'I64 targetF

              -- Choose starting endpoint based on direction
              vForwardS <- tbroadcast @'[] @s [] vForward
              qStart <- tselect vForwardS qp qm
              pStart <- tselect vForwardS pp pm

              -- Build subtree
              let Key kRestTensor = kRestKey
              Tuple4 qEnd pEnd packedSub kSub <-
                buildSubtree logpdf grad qStart pStart h0 u epsV targetSteps kRestTensor deltaMaxF

              -- Unpack subtree results
              nSub <- tslice1 packedSub 0
              sSubI64 <- tslice1 packedSub 1
              sSub <- equal sSubI64 oneI64

              -- Update candidate across subtrees with prob nSub / (nAcc + nSub)
              nTot <- tadd nAcc nSub
              nAccF <- convert @'[] @'I64 @'F32 nAcc
              nTotF <- convert @'[] @'I64 @'F32 nTot
              nSubF <- convert @'[] @'I64 @'F32 nSub
              prob  <- tdiv nSubF nTotF
              probD <- convert @'[] @'F32 @d prob
              (kCandKey, _) <- RNG.splitKey (Key kSub)
              uCand <- RNG.rngUniformF32 kCandKey >>= convert @'[] @'F32 @d
              acceptCand <- tlessThan uCand probD
              acceptCandS <- tbroadcast @'[] @s [] acceptCand
              qcNew <- tselect acceptCandS qEnd qc

              -- Update endpoints
              qmNew <- tselect vForwardS qm qEnd
              pmNew <- tselect vForwardS pm pEnd
              qpNew <- tselect vForwardS qEnd qp
              ppNew <- tselect vForwardS pEnd pp

              -- U-turn check
              noUTurn <- uTurnCheck qmNew pmNew qpNew ppNew
              sNew <- tand sSub noUTurn

              -- Update packed
              nNew <- tadd nAcc nSub
              sNewI64 <- tselect sNew oneI64 zeroI64
              jNew <- tadd jDep oneI64
              packedNew <- tpack3 nNew sNewI64 jNew

              let Key kCandTensor = kCandKey
              returnTuple7 qmNew pmNew qpNew ppNew qcNew packedNew kCandTensor)

      -- 5. Return new state
      let newPos = qcFin
      infoAcceptProb <- constant @'[] @d 1.0
      infoAccepted   <- constant @'[] @'Bool 1.0
      infoTreeDepth  <- tslice1 packedFin 2
      let info = Info infoAcceptProb infoAccepted infoTreeDepth
      return (NUTSState newPos p0 (nutsLogDens state) (nutsGradient state), info)

-----------------------------------------------------------------------------
-- Hamiltonian: H = K - logpdf = 0.5 * p^T p - logpdf(q)
-----------------------------------------------------------------------------

hamiltonian :: forall s d. (KnownShape s, KnownDType d)
            => (Tensor s d -> Builder (Tensor '[] d))
            -> Tensor s d -> Tensor s d
            -> Builder (Tensor '[] d)
hamiltonian logpdf pos mom = do
  ld <- logpdf pos
  kinetic <- do
    pSq  <- tmul mom mom
    pSum <- tsumAll pSq
    half <- constant @'[] @d 0.5
    tmul half pSum
  tsub kinetic ld

-----------------------------------------------------------------------------
-- U-turn condition: (q+ - q-) · p+ >= 0  AND  (q+ - q-) · p- >= 0
-----------------------------------------------------------------------------

uTurnCheck :: forall s d. (KnownShape s, KnownDType d)
           => Tensor s d -> Tensor s d -> Tensor s d -> Tensor s d
           -> Builder (Tensor '[] 'Bool)
uTurnCheck qm pm qp pp = do
  dq   <- tsub qp qm
  dotF <- tsumAll =<< tmul dq pp
  dotB <- tsumAll =<< tmul dq pm
  zeroD <- constant @'[] @d 0.0
  ge1  <- greaterThanOrEqual dotF zeroD
  ge2  <- greaterThanOrEqual dotB zeroD
  tand ge1 ge2

-----------------------------------------------------------------------------
-- Build subtree: iterative leapfrog integration
--
-- Carries: (pos, momentum, packed(n,s,count), key)
-- Returns: (q_end, p_end, packed(n,s,count), key)
--
-- The candidate is always the final state of the subtree.
-----------------------------------------------------------------------------

buildSubtree :: forall s d.
                (KnownShape s, KnownDType d)
             => (Tensor s d -> Builder (Tensor '[] d))   -- logpdf
             -> Gradient s d                             -- grad
             -> Tensor s d                               -- q_start
             -> Tensor s d                               -- p_start
             -> Tensor '[] d                             -- H0
             -> Tensor '[] d                             -- u (slice)
             -> Tensor '[] d                             -- eps_dir
             -> Tensor '[] 'I64                          -- target_steps
             -> Tensor '[2] 'UI64                        -- key tensor
             -> Double                                   -- deltaMax
             -> Builder (Tuple4 s d s d '[3] 'I64 '[2] 'UI64)
buildSubtree logpdf grad q0 p0 h0 u epsDir targetSteps key0 deltaMaxF = do
  zeroI64 <- constant @'[] @'I64 0
  oneI64  <- constant @'[] @'I64 1
  packed0 <- tpack3 zeroI64 oneI64 zeroI64   -- n=0, s=1, count=0

  Tuple4 qEnd pEnd packedEnd keyEnd <-
    whileLoop4 q0 p0 packed0 key0
      (\_ _ packed _ -> do
          sFlg <- tslice1 packed 1
          cnt  <- tslice1 packed 2
          sOk  <- equal sFlg oneI64
          cntOk <- lessThan cnt targetSteps
          tand sOk cntOk)
      (\q p packed k -> do
          g <- grad q

          -- Broadcast scalar step size to shape s for element-wise ops
          epsDirS <- tbroadcast @'[] @s [] epsDir
          half <- constant @'[] @d 0.5
          halfS <- tbroadcast @'[] @s [] half
          halfEpsS <- tmul epsDirS halfS

          -- Leapfrog step with eps_dir
          gScaled <- tmul g halfEpsS
          pHalf   <- tadd p gScaled
          pScaled <- tmul pHalf epsDirS
          q'      <- tadd q pScaled
          g'      <- grad q'
          gScaled' <- tmul g' halfEpsS
          p'      <- tadd pHalf gScaled'

          -- Hamiltonian and checks
          h' <- hamiltonian logpdf q' p'

          -- Divergence: H' - H0 > deltaMax
          delta    <- tsub h' h0
          deltaMaxT <- constant @'[] @d deltaMaxF
          diverged <- tlessThan deltaMaxT delta

          -- Slice: exp(-H') > u
          negH <- tnegate h'
          expNegH <- texp negH
          acceptable <- greaterThan expNegH u

          -- Update s: not diverged
          sNew <- tnot diverged

          -- Update n: add 1 if acceptable
          nAcc <- tslice1 packed 0
          nInc <- tselect acceptable oneI64 zeroI64
          nNew <- tadd nAcc nInc

          -- Update count
          cnt <- tslice1 packed 2
          cntNew <- tadd cnt oneI64

          -- Pack
          sNewI64 <- tselect sNew oneI64 zeroI64
          packedNew <- tpack3 nNew sNewI64 cntNew

          -- Split key for next iteration
          (Key kNext, _) <- RNG.splitKey (Key k)

          returnTuple4 q' p' packedNew kNext)

  returnTuple4 qEnd pEnd packedEnd keyEnd
