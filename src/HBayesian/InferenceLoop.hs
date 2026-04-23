{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

-- | Host-side inference loops for running compiled kernels.
--
-- Phase 2 uses a hybrid strategy: the kernel step is compiled to a PJRT
-- executable and executed repeatedly from Haskell IO. This avoids the
-- complexity of compiling the entire chain into a single XLA graph while
-- still running the heavy math on the device.
module HBayesian.InferenceLoop
  ( InferenceConfig (..)
  , defaultInferenceConfig
  , sampleChain
  ) where

import           Data.Int            (Int64)
import           Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)
import qualified HHLO.Runtime.Buffer as RTBuf
import qualified HHLO.Runtime.Compile as RT
import qualified HHLO.Runtime.Execute as RTExec
import qualified HHLO.Runtime.PJRT.Plugin as RT
import           HBayesian.Core
import           HBayesian.HHLO.Compile
import           HBayesian.HHLO.RNG ()

-- | Configuration for the inference loop.
data InferenceConfig = InferenceConfig
  { icNumWarmup   :: Int   -- ^ number of warmup (burn-in) steps
  , icNumSamples  :: Int   -- ^ number of sampling steps
  , icThinning    :: Int   -- ^ keep every N-th sample (1 = no thinning)
  }

defaultInferenceConfig :: InferenceConfig
defaultInferenceConfig = InferenceConfig
  { icNumWarmup  = 0
  , icNumSamples = 1000
  , icThinning   = 1
  }

-- | Run a single chain and collect samples.
--
-- This is a stub for Phase 2. A full implementation requires PJRT
-- plugin installation and buffer management. The function demonstrates
-- the intended API and will be completed when PJRT is available.
sampleChain :: forall s d state info.
               (KnownShape s, KnownDType d)
            => InferenceConfig
            -> Kernel s d state info
            -> Key                    -- ^ initial PRNG key
            -> Vector Double          -- ^ initial position (host)
            -> IO [Vector Double]     -- ^ collected samples (host)
sampleChain config kernel key0 pos0 = do
  -- NOTE: Full PJRT-based execution requires the PJRT CPU plugin.
  -- For Phase 2, this function demonstrates the API shape.
  -- When PJRT is available, the implementation will:
  --   1. Compile kernelInit and kernelStep to PJRT executables
  --   2. Transfer key and position to device buffers
  --   3. Run the warmup + sampling loop
  --   4. Read back samples to host vectors
  putStrLn "sampleChain: PJRT-based execution not yet enabled in Phase 2"
  return [pos0]
