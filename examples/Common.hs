{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Shared utilities for the Phase 2 example suite.
module Common
  ( renderKernelStep
  ) where

import           Data.Text           (Text)
import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)

-- | Render a single kernel step to StableHLO MLIR text.
renderKernelStep :: forall s d. (KnownShape s, KnownDType d)
                 => [FuncArg] -> Builder (Tensor s d) -> Text
renderKernelStep args b = render $ moduleFromBuilder @s @d "main" args b
