{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)
import           HBayesian.HHLO.Ops
import           HBayesian.HHLO.RNG

main :: IO ()
main = do
  putStrLn "=== HBayesian Phase 1 Demo ==="

  -- Demo 1: Render a simple uniform RNG program
  let uniformModule = render $ moduleFromBuilder @'[4] @'F32 "uniform_demo"
        [ FuncArg "key" (TensorType [2] UI64) ] $ do
          k <- arg @'[2] @'UI64
          rngUniformF32 (Key k)
  putStrLn "\n-- Uniform RNG MLIR --"
  putStrLn (show uniformModule)

  -- Demo 2: Render a normal RNG program
  let normalModule = render $ moduleFromBuilder @'[4] @'F32 "normal_demo"
        [ FuncArg "key" (TensorType [2] UI64) ] $ do
          k <- arg @'[2] @'UI64
          rngNormalF32 (Key k)
  putStrLn "\n-- Normal RNG MLIR --"
  putStrLn (show normalModule)

  putStrLn "\n=== Demo complete ==="
