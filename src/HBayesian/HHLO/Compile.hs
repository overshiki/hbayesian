{-# LANGUAGE OverloadedStrings #-}

-- | Compilation and execution utilities for HBayesian kernels.
--
-- This module provides thin wrappers around HHLO's PJRT runtime.
-- In Phase 1 it is intentionally minimal; it will expand as the project
-- matures.
module HBayesian.HHLO.Compile
  ( compileModule
  , renderBuilder
  ) where

import qualified Data.Text              as T
import           HHLO.IR.AST            (Module)
import           HHLO.IR.Builder        (Builder)
import           HHLO.IR.Pretty         (render)
import qualified HHLO.Runtime.Compile      as RT
import qualified HHLO.Runtime.PJRT.Plugin  as RT
import           HHLO.Runtime.PJRT.Types   (PJRTApi, PJRTClient, PJRTExecutable)

-- | Render a 'Builder' action to its StableHLO MLIR text.
--
-- This is primarily useful for debugging and golden tests.
renderBuilder :: Builder a -> Module -> T.Text
renderBuilder _ = render

-- | Compile a 'Module' to a PJRT executable.
--
-- Requires an active PJRT API and client. Use 'RT.withPJRT' to obtain them.
compileModule :: PJRTApi -> PJRTClient -> Module -> IO PJRTExecutable
compileModule api client modl =
  RT.compileWithOptions api client (render modl) RT.defaultCompileOptions
