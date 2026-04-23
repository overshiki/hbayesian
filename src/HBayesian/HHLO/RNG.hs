{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

module HBayesian.HHLO.RNG
  ( -- Re-export Key from Core so all modules use the same type
    Key (..)
  , splitKey
  , rngUniformF32
  , rngUniformF64
  , rngNormalF32
  , rngNormalF64
  , rngBernoulli
  ) where

import           Data.Word           (Word64)
import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.Builder
import           HBayesian.Core      (Key (..))
import           HBayesian.HHLO.Ops

-----------------------------------------------------------------------------
-- Key splitting
-----------------------------------------------------------------------------

splitKey :: Key -> Builder (Key, Key)
splitKey (Key k) = do
  (k1, _) <- rngBitGenerator @'[2] k
  (k2, _) <- rngBitGenerator @'[2] k1
  return (Key k1, Key k2)

-----------------------------------------------------------------------------
-- Uniform [0,1)
-----------------------------------------------------------------------------

rngUniformF32 :: forall s. KnownShape s => Key -> Builder (Tensor s 'F32)
rngUniformF32 (Key k) = do
  (_, bits) <- rngBitGenerator @s k
  bitsF32 <- convert bits
  maxVal <- constant @'[] @'F32 (fromIntegral (maxBound :: Word64))
  maxValBC <- broadcastWithDims @'[] @s [] maxVal
  tdiv bitsF32 maxValBC

rngUniformF64 :: forall s. KnownShape s => Key -> Builder (Tensor s 'F64)
rngUniformF64 (Key k) = do
  (_, bits) <- rngBitGenerator @s k
  bitsF64 <- convert bits
  maxVal <- constant @'[] @'F64 (fromIntegral (maxBound :: Word64))
  maxValBC <- broadcastWithDims @'[] @s [] maxVal
  tdiv bitsF64 maxValBC

-----------------------------------------------------------------------------
-- Standard normal (Box-Muller)
-----------------------------------------------------------------------------

rngNormalF32 :: forall s. KnownShape s => Key -> Builder (Tensor s 'F32)
rngNormalF32 key = do
  (key1, key2) <- splitKey key
  u1 <- rngUniformF32 key1
  u2 <- rngUniformF32 key2
  twoPi <- constant @'[] @'F32 (2.0 * pi)
  negTwo <- constant @'[] @'F32 (-2.0)
  twoPiBC <- broadcastWithDims @'[] @s [] twoPi
  negTwoBC <- broadcastWithDims @'[] @s [] negTwo
  logU1 <- tlog u1
  term1 <- tmul negTwoBC logU1
  sqrtTerm1 <- tsqrt term1
  angle <- tmul twoPiBC u2
  cosAngle <- tcos angle
  tmul sqrtTerm1 cosAngle

rngNormalF64 :: forall s. KnownShape s => Key -> Builder (Tensor s 'F64)
rngNormalF64 key = do
  (key1, key2) <- splitKey key
  u1 <- rngUniformF64 key1
  u2 <- rngUniformF64 key2
  twoPi <- constant @'[] @'F64 (2.0 * pi)
  negTwo <- constant @'[] @'F64 (-2.0)
  twoPiBC <- broadcastWithDims @'[] @s [] twoPi
  negTwoBC <- broadcastWithDims @'[] @s [] negTwo
  logU1 <- tlog u1
  term1 <- tmul negTwoBC logU1
  sqrtTerm1 <- tsqrt term1
  angle <- tmul twoPiBC u2
  cosAngle <- tcos angle
  tmul sqrtTerm1 cosAngle

-----------------------------------------------------------------------------
-- Bernoulli
-----------------------------------------------------------------------------

rngBernoulli :: forall s. KnownShape s => Key -> Tensor s 'F32 -> Builder (Tensor s 'Bool)
rngBernoulli key probs = do
  u <- rngUniformF32 key
  lessThanEW u probs
