{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.HHLO.RNG (tests) where

import           Data.Text           (Text)
import qualified Data.Text           as T
import           Test.Tasty
import           Test.Tasty.HUnit

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)
import           HBayesian.HHLO.Ops
import           HBayesian.HHLO.RNG

render1 :: forall s d. (KnownShape s, KnownDType d)
        => [FuncArg] -> Builder (Tensor s d) -> Text
render1 args b = render $ moduleFromBuilder @s @d "main" args b

render2 :: forall s1 d1 s2 d2. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
        => [FuncArg] -> Builder (Tuple2 s1 d1 s2 d2) -> Text
render2 args b = render $ moduleFromBuilder2 @s1 @d1 @s2 @d2 "main" args b

tests :: TestTree
tests = testGroup "HHLO.RNG"
  [ testCase "splitKey renders two rng_bit_generator ops" $ do
      let mlir = render2 @'[2] @'UI64 @'[2] @'UI64
                   [ FuncArg "key" (TensorType [2] UI64) ] $ do
            k <- arg @'[2] @'UI64
            (k1, k2) <- splitKey (Key k)
            returnTuple2 (unKey k1) (unKey k2)
      -- We just check that the expected ops appear
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains THREE_FRY" (T.isInfixOf "THREE_FRY" mlir)

  , testCase "rngUniformF32 renders" $ do
      let mlir = render1 @'[3] @'F32
                   [ FuncArg "key" (TensorType [2] UI64) ] $ do
            k <- arg @'[2] @'UI64
            rngUniformF32 (Key k)
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains convert" (T.isInfixOf "stablehlo.convert" mlir)
      assertBool "contains divide" (T.isInfixOf "stablehlo.divide" mlir)

  , testCase "rngNormalF32 renders" $ do
      let mlir = render1 @'[3] @'F32
                   [ FuncArg "key" (TensorType [2] UI64) ] $ do
            k <- arg @'[2] @'UI64
            rngNormalF32 (Key k)
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains log" (T.isInfixOf "stablehlo.log" mlir)
      assertBool "contains sqrt" (T.isInfixOf "stablehlo.sqrt" mlir)
      assertBool "contains cosine" (T.isInfixOf "stablehlo.cosine" mlir)
  ]
