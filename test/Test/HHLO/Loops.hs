{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.HHLO.Loops (tests) where

import           Data.Text           (Text)
import qualified Data.Text           as T
import           Test.Tasty
import           Test.Tasty.HUnit

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.AST         (FuncArg(..), TensorType(..), Module(..))
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)
import           HBayesian.HHLO.Ops
import           HBayesian.HHLO.Loops

-- | Build a 3-tuple and render it.
render3T :: forall s1 d1 s2 d2 s3 d3.
            (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3)
         => [FuncArg] -> Builder (Tuple '[s1, s2, s3] '[d1, d2, d3]) -> Text
render3T args b = render $ moduleFromBuilderT @'[s1, s2, s3] @'[d1, d2, d3] "main" args b

tests :: TestTree
tests = testGroup "HHLO.Loops"
  [ testCase "whileLoop3 renders stablehlo.while" $ do
      let mlir = render3T @'[2] @'F32 @'[2] @'F32 @'[] @'I64
                   [ FuncArg "a" (TensorType [2] F32)
                   , FuncArg "b" (TensorType [2] F32)
                   , FuncArg "c" (TensorType [] I64)
                   ] $ do
            a0 <- arg @'[2] @'F32
            b0 <- arg @'[2] @'F32
            c0 <- arg @'[] @'I64
            (a1, b1, c1) <- whileLoop3 a0 b0 c0
              (\a b c -> do
                limit <- constant @'[] @'I64 10
                lessThan c limit)
              (\a b c -> do
                a' <- tadd a b
                b' <- tmul b a
                one <- constant @'[] @'I64 1
                c' <- add c one
                return (a', b', c'))
            return $ a1 ::: b1 ::: c1 ::: TNil
      assertBool "contains stablehlo.while" (T.isInfixOf "stablehlo.while" mlir)
      assertBool "contains stablehlo.return" (T.isInfixOf "stablehlo.return" mlir)

  , testCase "conditional3 renders stablehlo.if" $ do
      let mlir = render3T @'[2] @'F32 @'[2] @'F32 @'[] @'I64
                   [ FuncArg "p" (TensorType [] Bool)
                   ] $ do
            p <- arg @'[] @'Bool
            (a, b, c) <- conditional3 p
              (do
                x <- constant @'[2] @'F32 1.0
                y <- constant @'[2] @'F32 2.0
                z <- constant @'[] @'I64 3
                return (x, y, z))
              (do
                x <- constant @'[2] @'F32 0.0
                y <- constant @'[2] @'F32 0.0
                z <- constant @'[] @'I64 0
                return (x, y, z))
            return $ a ::: b ::: c ::: TNil
      assertBool "contains stablehlo.if" (T.isInfixOf "stablehlo.if" mlir)
  ]
