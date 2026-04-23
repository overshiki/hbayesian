{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}

-- | Core types and abstractions for HBayesian.
module HBayesian.Core
  ( -- * PRNG
    Key (..)
    -- * Gradients
  , Gradient
    -- * State and diagnostics
  , State (..)
  , Info (..)
    -- * Kernel abstraction
  , Kernel (..)
  , SimpleKernel
  ) where

import HHLO.Core.Types (DType (..), Shape)
import HHLO.IR.Builder (Builder, Tensor)

--------------------------------------------------------------------------------
-- PRNG
--------------------------------------------------------------------------------

-- | A functional PRNG key backed by Threefry.
newtype Key = Key { unKey :: Tensor '[2] 'UI64 }

--------------------------------------------------------------------------------
-- Gradients
--------------------------------------------------------------------------------

-- | A user-provided gradient function.
type Gradient (s :: Shape) (d :: DType) = Tensor s d -> Builder (Tensor s d)

--------------------------------------------------------------------------------
-- State
--------------------------------------------------------------------------------

-- | Minimal algorithm state: position and log-density.
data State (s :: Shape) (d :: DType) = State
  { statePosition   :: !(Tensor s d)
  , stateLogDensity :: !(Tensor '[] d)
  }

--------------------------------------------------------------------------------
-- Diagnostics
--------------------------------------------------------------------------------

-- | Diagnostic information produced by a single transition.
data Info (s :: Shape) (d :: DType) = Info
  { infoAcceptProb :: !(Tensor '[] d)
  , infoAccepted   :: !(Tensor '[] 'Bool)
  , infoNumSteps   :: !(Tensor '[] 'I64)
  }

--------------------------------------------------------------------------------
-- Kernel
--------------------------------------------------------------------------------

-- | A transition kernel, polymorphic in state and info.
--
-- Algorithms define their own state and info types, then expose a
-- 'Kernel s d state info'. The convenience alias 'SimpleKernel' covers
-- the common gradient-free case.
data Kernel (s :: Shape) (d :: DType) state info = Kernel
  { kernelInit :: !(Key -> Tensor s d -> Builder state)
  , kernelStep :: !(Key -> state -> Builder (state, info))
  }

-- | A kernel that uses the base 'State' and 'Info' types.
type SimpleKernel s d = Kernel s d (State s d) (Info s d)
