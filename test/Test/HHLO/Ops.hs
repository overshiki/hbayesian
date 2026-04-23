{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.HHLO.Ops (tests) where

import           Data.Text           (Text)
import           Test.Tasty
import           Test.Tasty.HUnit

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)
import           HBayesian.HHLO.Ops

-- | Render a single-result builder to text.
render1 :: forall s d. (KnownShape s, KnownDType d)
        => [FuncArg] -> Builder (Tensor s d) -> Text
render1 args b = render $ moduleFromBuilder @s @d "main" args b

tests :: TestTree
tests = testGroup "HHLO.Ops"
  [ testCase "sqrt op renders" $ do
      let mlir = render1 @'[2,2] @'F32
                   [ FuncArg "x" (TensorType [2,2] F32) ] $ do
            x <- arg @'[2,2] @'F32
            sqrt' x
      let expected = "module {\n  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {\n      %0 = stablehlo.sqrt %arg0 : (tensor<2x2xf32>) -> tensor<2x2xf32>\n      return %0 : tensor<2x2xf32>\n  }\n}"
      mlir @?= expected

  , testCase "sin op renders" $ do
      let mlir = render1 @'[3] @'F64
                   [ FuncArg "x" (TensorType [3] F64) ] $ do
            x <- arg @'[3] @'F64
            sin' x
      let expected = "module {\n  func.func @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {\n      %0 = stablehlo.sine %arg0 : (tensor<3xf64>) -> tensor<3xf64>\n      return %0 : tensor<3xf64>\n  }\n}"
      mlir @?= expected

  , testCase "cos op renders" $ do
      let mlir = render1 @'[3] @'F64
                   [ FuncArg "x" (TensorType [3] F64) ] $ do
            x <- arg @'[3] @'F64
            cos' x
      let expected = "module {\n  func.func @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {\n      %0 = stablehlo.cosine %arg0 : (tensor<3xf64>) -> tensor<3xf64>\n      return %0 : tensor<3xf64>\n  }\n}"
      mlir @?= expected

  , testCase "pow op renders" $ do
      let mlir = render1 @'[2] @'F32
                   [ FuncArg "x" (TensorType [2] F32)
                   , FuncArg "y" (TensorType [2] F32)
                   ] $ do
            x <- arg @'[2] @'F32
            y <- arg @'[2] @'F32
            pow' x y
      let expected = "module {\n  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {\n      %0 = stablehlo.power %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>\n      return %0 : tensor<2xf32>\n  }\n}"
      mlir @?= expected

  , testCase "element-wise lessThan renders" $ do
      let mlir = render1 @'[2] @'Bool
                   [ FuncArg "x" (TensorType [2] F32)
                   , FuncArg "y" (TensorType [2] F32)
                   ] $ do
            x <- arg @'[2] @'F32
            y <- arg @'[2] @'F32
            lessThanEW x y
      let expected = "module {\n  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xi1> {\n      %0 = \"stablehlo.compare\"(%arg0, %arg1) {comparison_direction = \"LT\"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>\n      return %0 : tensor<2xi1>\n  }\n}"
      mlir @?= expected
  ]
