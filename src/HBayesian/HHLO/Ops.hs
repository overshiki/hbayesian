{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module HBayesian.HHLO.Ops
  ( module HHLO.EDSL.Ops
  , module HHLO.Core.Types
    -- * Missing primitive ops
  , sqrt'
  , rsqrt'
  , sin'
  , cos'
  , tan'
  , pow'
  , log1p'
  , floor'
  , ceil'
    -- * Element-wise comparison (returns tensor of bools)
  , lessThanEW
  , greaterThanEW
  , equalEW
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
import           Data.Proxy
import           Data.Text       (Text)
import qualified Data.Text as T
import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.AST         (Attribute (..))
import           HHLO.IR.Builder
import           Prelude hiding (negate, minimum, maximum)

-----------------------------------------------------------------------------
-- Missing primitive ops (not in HHLO 0.2.0.0)
-----------------------------------------------------------------------------

sqrt' :: forall s d. (KnownShape s, KnownDType d)
      => Tensor s d -> Builder (Tensor s d)
sqrt' (Tensor x) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.sqrt" [x] [ttype] [] ttype
  return (Tensor vid)

rsqrt' :: forall s d. (KnownShape s, KnownDType d)
       => Tensor s d -> Builder (Tensor s d)
rsqrt' (Tensor x) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.rsqrt" [x] [ttype] [] ttype
  return (Tensor vid)

sin' :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
sin' (Tensor x) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.sine" [x] [ttype] [] ttype
  return (Tensor vid)

cos' :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
cos' (Tensor x) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.cosine" [x] [ttype] [] ttype
  return (Tensor vid)

tan' :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
tan' (Tensor x) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.tangent" [x] [ttype] [] ttype
  return (Tensor vid)

pow' :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Tensor s d -> Builder (Tensor s d)
pow' (Tensor x) (Tensor y) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.power" [x, y] [ttype, ttype] [] ttype
  return (Tensor vid)

log1p' :: forall s d. (KnownShape s, KnownDType d)
       => Tensor s d -> Builder (Tensor s d)
log1p' (Tensor x) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.log_plus_one" [x] [ttype] [] ttype
  return (Tensor vid)

floor' :: forall s d. (KnownShape s, KnownDType d)
       => Tensor s d -> Builder (Tensor s d)
floor' (Tensor x) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.floor" [x] [ttype] [] ttype
  return (Tensor vid)

ceil' :: forall s d. (KnownShape s, KnownDType d)
      => Tensor s d -> Builder (Tensor s d)
ceil' (Tensor x) = do
  let ttype = tensorType (Proxy @s) (Proxy @d)
  vid <- emitOp "stablehlo.ceil" [x] [ttype] [] ttype
  return (Tensor vid)

-----------------------------------------------------------------------------
-- Element-wise comparison (HHLO's compare returns scalar; we fix it)
-----------------------------------------------------------------------------

lessThanEW :: forall s d. (KnownShape s, KnownDType d)
           => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
lessThanEW (Tensor x) (Tensor y) = do
  let inType  = tensorType (Proxy @s) (Proxy @d)
      outType = tensorType (Proxy @s) (Proxy @'Bool)
  vid <- emitOp "stablehlo.compare" [x, y] [inType, inType]
           [AttrString "comparison_direction" "LT"] outType
  return (Tensor vid)

greaterThanEW :: forall s d. (KnownShape s, KnownDType d)
              => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
greaterThanEW (Tensor x) (Tensor y) = do
  let inType  = tensorType (Proxy @s) (Proxy @d)
      outType = tensorType (Proxy @s) (Proxy @'Bool)
  vid <- emitOp "stablehlo.compare" [x, y] [inType, inType]
           [AttrString "comparison_direction" "GT"] outType
  return (Tensor vid)

equalEW :: forall s d. (KnownShape s, KnownDType d)
        => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
equalEW (Tensor x) (Tensor y) = do
  let inType  = tensorType (Proxy @s) (Proxy @d)
      outType = tensorType (Proxy @s) (Proxy @'Bool)
  vid <- emitOp "stablehlo.compare" [x, y] [inType, inType]
           [AttrString "comparison_direction" "EQ"] outType
  return (Tensor vid)

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
tsqrt = sqrt'

trsqrt :: forall s d. (KnownShape s, KnownDType d)
       => Tensor s d -> Builder (Tensor s d)
trsqrt = rsqrt'

tsin :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
tsin = sin'

tcos :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
tcos = cos'

ttan :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Builder (Tensor s d)
ttan = tan'

tpow :: forall s d. (KnownShape s, KnownDType d)
     => Tensor s d -> Tensor s d -> Builder (Tensor s d)
tpow = pow'

tlessThan :: forall s d. (KnownShape s, KnownDType d)
          => Tensor s d -> Tensor s d -> Builder (Tensor '[] 'Bool)
tlessThan = lessThan

tselect :: forall s d. (KnownShape s, KnownDType d)
        => Tensor s 'Bool -> Tensor s d -> Tensor s d -> Builder (Tensor s d)
tselect = select

tconstant :: forall s d. (KnownShape s, KnownDType d)
          => Double -> Builder (Tensor s d)
tconstant = constant

tsumAll :: forall s d. (KnownShape s, KnownDType d)
        => Tensor s d -> Builder (Tensor '[] d)
tsumAll = reduceSum

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
tslice1 vec i = do
  sliced <- slice @'[n] @'[1] @d vec [i] [i+1] [1]
  treshape @'[1] @'[] sliced

-- | Pack two scalar tensors into a rank-1 tensor of shape [2].
tpack2 :: forall d. (KnownDType d)
       => Tensor '[] d -> Tensor '[] d -> Builder (Tensor '[2] d)
tpack2 x y = do
  x1 <- treshape @'[] @'[1] x
  y1 <- treshape @'[] @'[1] y
  concatenate @'[1] @'[2] @d 0 [x1, y1]

-- | Pack three scalar tensors into a rank-1 tensor of shape [3].
tpack3 :: forall d. (KnownDType d)
       => Tensor '[] d -> Tensor '[] d -> Tensor '[] d -> Builder (Tensor '[3] d)
tpack3 x y z = do
  x1 <- treshape @'[] @'[1] x
  y1 <- treshape @'[] @'[1] y
  z1 <- treshape @'[] @'[1] z
  concatenate @'[1] @'[3] @d 0 [x1, y1, z1]

-----------------------------------------------------------------------------
-- Sigmoid and matmul
-----------------------------------------------------------------------------

-- | Element-wise sigmoid: 1 / (1 + exp(-x)).
tsigmoid :: forall s. (KnownShape s)
         => Tensor s 'F32 -> Builder (Tensor s 'F32)
tsigmoid x = do
  negX <- tnegate x
  expNegX <- texp negX
  one <- tconstant @s @'F32 1.0
  denom <- tadd one expNegX
  tdiv one denom


