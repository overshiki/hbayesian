{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Shared utilities for the Phase 2 example suite.
module Common
  ( -- * Tier A: MLIR rendering
    renderKernelStep
    -- * Tier B: PJRT execution helpers
  , compileModule
  , executeModule
  , bufferFromF32
  , bufferFromUI64
  , bufferToF32
  , bufferToUI64
  ) where

import           Data.Int            (Int64)
import           Data.Text           (Text)
import           Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import           Data.Word           (Word64)
import           Foreign.C.Types     (CInt)

import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..), Module)
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)
import qualified Data.Text as T
import           HHLO.Runtime.Buffer  (toDevice, toDeviceF32, fromDevice, fromDeviceF32)
import           HHLO.Runtime.Compile (compileWithOptions, defaultCompileOptions)
import           HHLO.Runtime.Execute (execute)
import           HHLO.Runtime.PJRT.Types (PJRTApi, PJRTClient, PJRTExecutable, PJRTBuffer,
                                           bufferTypeU64)

-----------------------------------------------------------------------------
-- Tier A: MLIR rendering
-----------------------------------------------------------------------------

renderKernelStep :: forall s d. (KnownShape s, KnownDType d)
                 => [FuncArg] -> Builder (Tensor s d) -> Text
renderKernelStep args b = render $ moduleFromBuilder @s @d "main" args b

-----------------------------------------------------------------------------
-- Tier B: PJRT execution
-----------------------------------------------------------------------------

-- | Compile a StableHLO 'Module' to a PJRT executable.
compileModule :: PJRTApi -> PJRTClient -> Module -> IO PJRTExecutable
compileModule api client modl =
  compileWithOptions api client (render modl) defaultCompileOptions

-- | Execute a compiled PJRT executable with input buffers.
executeModule :: PJRTApi -> PJRTExecutable -> [PJRTBuffer] -> IO [PJRTBuffer]
executeModule = execute

-- | Create a device buffer from a list of 'Float' values.
bufferFromF32 :: PJRTApi -> PJRTClient -> [Int] -> [Float] -> IO PJRTBuffer
bufferFromF32 api client dims vals =
  toDeviceF32 api client (V.fromList vals) (map fromIntegral dims)

-- | Create a device buffer from a list of 'Word64' values.
bufferFromUI64 :: PJRTApi -> PJRTClient -> [Int] -> [Word64] -> IO PJRTBuffer
bufferFromUI64 api client dims vals =
  toDevice api client (V.fromList vals) (map fromIntegral dims) bufferTypeU64

-- | Read back 'Float' values from a device buffer.
bufferToF32 :: PJRTApi -> PJRTBuffer -> Int -> IO [Float]
bufferToF32 api buf n = V.toList <$> fromDeviceF32 api buf n

-- | Read back 'Word64' values from a device buffer.
bufferToUI64 :: PJRTApi -> PJRTBuffer -> Int -> IO [Word64]
bufferToUI64 api buf n = V.toList <$> fromDevice api buf n
