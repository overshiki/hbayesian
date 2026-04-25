{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module HBayesian.HHLO.Ops
  ( module HHLO.EDSL.Ops
  , module HHLO.Core.Types
    -- * Convenience aliases
  , tadd
  , tsub
  , tmul
  , tdiv
  , tnegate
  , tabs
  , texp
  , tlog
  , tsqrt
  , trsqrt
  , tsin
  , tcos
  , ttan
  , tpow
  , tlessThan
  , tminimum
  , tmaximum
  , tselect
  , tconstant
  , tsumAll
  , treshape
  , tbroadcast
  , tslice1
  , tpack2
  , tpack3
  , tsigmoid
  ) where

import           Data.Int        (Int64)
import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.Builder
import           Prelude hiding (negate, minimum, maximum, sqrt, sin, cos, tan, floor, ceiling)

-----------------------------------------------------------------------------
-- Convenience aliases
-----------------------------------------------------------------------------

tadd, tsub, tmul, tdiv :: forall s d.
  (KnownShape s, KnownDType d)
  => Tensor s d -> Tensor s d -> Builder (Tensor s d)
tadd = add
tsub = sub
tmul = multiply
tdiv = divide

tnegate :: forall s d. (KnownShape s, KnownDType d)
        => Tensor s d -> Builder (Tensor s d)
tnegate = negate

tabs :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
tabs = abs'

texp :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
texp = exponential

tlog :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
tlog = logarithm

tsqrt :: forall s d. (KnownShape s, KnownDType d)
      => Tensor s d -> Builder (Tensor s d)
tsqrt = sqrt

trsqrt :: forall s d. (KnownShape s, KnownDType d)
       => Tensor s d -> Builder (Tensor s d)
trsqrt = rsqrt

tsin :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
tsin = sin

tcos :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
tcos = cos

ttan :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
ttan = tan

tpow :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Tensor s d -> Builder (Tensor s d)
tpow = pow

tlessThan :: forall s d. (KnownShape s, KnownDType d)
          => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
tlessThan = lessThan

tselect :: forall s d. (KnownShape s, KnownDType d)
        => Tensor s 'Bool -> Tensor s d -> Tensor s d -> Builder (Tensor s d)
tselect = select

tconstant :: forall s d. (KnownShape s, KnownDType d)
          => Double -> Builder (Tensor s d)
tconstant = constant

tsumAll :: forall s d. (KnownShape s, KnownDType d)
        => Tensor s d -> Builder (Tensor '[] d)
tsumAll = sumAll

treshape :: forall sFrom sTo d.
  (KnownShape sFrom, KnownShape sTo, KnownDType d)
  => Tensor sFrom d -> Builder (Tensor sTo d)
treshape = reshape

tbroadcast :: forall sFrom sTo d.
  (KnownShape sFrom, KnownShape sTo, KnownDType d)
  => [Int64] -> Tensor sFrom d -> Builder (Tensor sTo d)
tbroadcast = broadcastWithDims

tminimum :: forall s d. (KnownShape s, KnownDType d)
         => Tensor s d -> Tensor s d -> Builder (Tensor s d)
tminimum = minimum

tmaximum :: forall s d. (KnownShape s, KnownDType d)
         => Tensor s d -> Tensor s d -> Builder (Tensor s d)
tmaximum = maximum

-----------------------------------------------------------------------------
-- Slice / pack helpers for parameter vectors
-----------------------------------------------------------------------------

-- | Extract a single scalar element from a 1-D tensor at a constant index.
tslice1 :: forall n d. (KnownShape '[n], KnownDType d)
        => Tensor '[n] d -> Int64 -> Builder (Tensor '[] d)
tslice1 = slice1

-- | Pack two scalar tensors into a rank-1 tensor of shape [2].
tpack2 :: forall d. (KnownDType d)
       => Tensor '[] d -> Tensor '[] d -> Builder (Tensor '[2] d)
tpack2 = pack2

-- | Pack three scalar tensors into a rank-1 tensor of shape [3].
tpack3 :: forall d. (KnownDType d)
       => Tensor '[] d -> Tensor '[] d -> Tensor '[] d -> Builder (Tensor '[3] d)
tpack3 = pack3

-----------------------------------------------------------------------------
-- Sigmoid
-----------------------------------------------------------------------------

-- | Element-wise sigmoid: 1 / (1 + exp(-x)).
tsigmoid :: forall s. (KnownShape s)
         => Tensor s 'F32 -> Builder (Tensor s 'F32)
tsigmoid = sigmoid
