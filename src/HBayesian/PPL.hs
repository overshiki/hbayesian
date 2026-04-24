{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

-- | A shallow probabilistic programming layer for HBayesian.
--
-- The PPL is a monad that carries:
--   * a read-only parameter vector of shape @[n]@
--   * a mutable log-density accumulator (scalar)
--
-- It desugars to the same 'Tensor s d -> Builder (Tensor '[] d)'
-- that samplers expect.
--
-- Example:
--
-- > myModel :: PPL 2 ()
-- > myModel = do
-- >     alpha <- param 0
-- >     beta  <- param 1
-- >     observe "alpha_prior" (normal 0.0 1.0) alpha
-- >     observe "beta_prior"  (normal 0.0 1.0) beta
-- >     forM_ dataset $ \(x, y) -> do
-- >         let mu = alpha + beta * x
-- >         observe "y" (normal mu 0.5) y
-- >
-- > logpdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
-- > logpdf = runPPL myModel
module HBayesian.PPL
  ( -- * PPL monad
    PPL
  , runPPL
  , liftBuilder
    -- * Model vocabulary
  , param
  , observe
    -- * Distribution primitives
  , normal
  , normalT
  , uniform
  , uniformT
  , halfNormal
  , bernoulli
  , bernoulliT
  ) where

import           GHC.TypeNats        (Nat)

import           HHLO.Core.Types
import           HHLO.IR.Builder
import           HBayesian.HHLO.Ops hiding (map)
import qualified HHLO.EDSL.Ops as EDSL

-----------------------------------------------------------------------------
-- PPL monad (manual implementation, no mtl dependency)
-----------------------------------------------------------------------------

-- | The PPL monad. Carries:
--   * a read-only parameter vector of shape @[n]@
--   * a mutable log-density accumulator (scalar)
newtype PPL (n :: Nat) a = PPL
    { unPPL :: Tensor '[n] 'F32 -> Tensor '[] 'F32 -> Builder (a, Tensor '[] 'F32)
    }

instance Functor (PPL n) where
    fmap f (PPL m) = PPL $ \theta acc -> do
        (x, acc') <- m theta acc
        return (f x, acc')

instance Applicative (PPL n) where
    pure x = PPL $ \_ acc -> return (x, acc)
    PPL mf <*> PPL mx = PPL $ \theta acc -> do
        (f, acc1) <- mf theta acc
        (x, acc2) <- mx theta acc1
        return (f x, acc2)

instance Monad (PPL n) where
    return = pure
    PPL mx >>= f = PPL $ \theta acc -> do
        (x, acc1) <- mx theta acc
        unPPL (f x) theta acc1

-- | Lift a raw 'Builder' action into the PPL.
liftBuilder :: Builder a -> PPL n a
liftBuilder mx = PPL $ \_ acc -> do
    x <- mx
    return (x, acc)

-- | Run a PPL model starting from a parameter vector and zero log-density.
-- Returns the accumulated log-posterior.
runPPL :: KnownShape '[n] => PPL n () -> Tensor '[n] 'F32 -> Builder (Tensor '[] 'F32)
runPPL (PPL m) theta = do
    zero <- tconstant 0.0
    ((), acc) <- m theta zero
    return acc

-----------------------------------------------------------------------------
-- Model vocabulary
-----------------------------------------------------------------------------

-- | Extract the i-th scalar parameter from the parameter vector.
param :: forall n. (KnownShape '[n], KnownDType 'F32)
      => Int -> PPL n (Tensor '[] 'F32)
param i = PPL $ \theta acc -> do
    x <- tslice1 @n @'F32 theta (fromIntegral i)
    return (x, acc)

-- | Condition the model on an observed value from a distribution.
-- The distribution is a function @value -> Builder (Tensor '[] 'F32)@
-- that computes the log-density.
observe :: String -> (Tensor '[] 'F32 -> Builder (Tensor '[] 'F32)) -> Tensor '[] 'F32 -> PPL n ()
observe _name dist value = PPL $ \_ acc -> do
    ld <- dist value
    newAcc <- tadd acc ld
    return ((), newAcc)

-----------------------------------------------------------------------------
-- Distribution primitives
-----------------------------------------------------------------------------

-- | Normal distribution with constant mean and std (unnormalised log-pdf).
normal :: Double -> Double -> Tensor '[] 'F32 -> Builder (Tensor '[] 'F32)
normal mean std x = do
    meanT <- tconstant mean
    stdT  <- tconstant std
    normalT meanT stdT x

-- | Normal distribution with tensor mean and std.
normalT :: Tensor '[] 'F32 -> Tensor '[] 'F32 -> Tensor '[] 'F32 -> Builder (Tensor '[] 'F32)
normalT meanT stdT x = do
    diff  <- tsub x meanT
    ratio <- tdiv diff stdT
    sq    <- tmul ratio ratio
    negHalf <- tconstant (-0.5)
    tmul negHalf sq

-- | Uniform distribution on @[a, b]@ with constant bounds.
uniform :: Double -> Double -> Tensor '[] 'F32 -> Builder (Tensor '[] 'F32)
uniform a b x = do
    aT <- tconstant a
    bT <- tconstant b
    uniformT aT bT x

-- | Uniform distribution with tensor bounds.
uniformT :: Tensor '[] 'F32 -> Tensor '[] 'F32 -> Tensor '[] 'F32 -> Builder (Tensor '[] 'F32)
uniformT aT bT x = do
    geA <- EDSL.compare x aT "GE"
    leB <- EDSL.compare x bT "LE"
    zero <- tconstant 0.0
    negInf <- tconstant (-1.0e30)  -- proxy for -inf
    -- inside = geA && leB  via nested tselect
    inner <- tselect leB zero negInf
    tselect geA inner negInf

-- | Half-normal distribution (positive support) with constant std.
halfNormal :: Double -> Tensor '[] 'F32 -> Builder (Tensor '[] 'F32)
halfNormal std x = do
    zero <- tconstant 0.0
    isPos <- EDSL.compare x zero "GE"
    stdT <- tconstant std
    lp <- normalT zero stdT x
    negInf <- tconstant (-1.0e30)
    tselect isPos lp negInf

-- | Bernoulli distribution with constant probability.
bernoulli :: Double -> Tensor '[] 'F32 -> Builder (Tensor '[] 'F32)
bernoulli p y = do
    pT <- tconstant p
    bernoulliT pT y

-- | Bernoulli distribution with tensor probability.
bernoulliT :: Tensor '[] 'F32 -> Tensor '[] 'F32 -> Builder (Tensor '[] 'F32)
bernoulliT pT y = do
    logP   <- tlog pT
    one    <- tconstant 1.0
    oneMinP <- tsub one pT
    logOneMinP <- tlog oneMinP
    term1  <- tmul y logP
    oneMinY <- tsub one y
    term2  <- tmul oneMinY logOneMinP
    tadd term1 term2
