{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Examples (tests) where

import           Data.Text           (Text)
import qualified Data.Text           as T
import           Test.Tasty
import           Test.Tasty.HUnit

import qualified LinearRegressionRandomWalk as Ex1
import qualified GaussianProcessEllipticalSlice as Ex2
import qualified LogisticRegressionHMC as Ex3
import qualified BivariateGaussianMALA as Ex4

tests :: TestTree
tests = testGroup "Examples"
  [ testCase "LinearRegressionRandomWalk renders MLIR" $ do
      let mlir = Ex1.renderStepMlir
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains add" (T.isInfixOf "stablehlo.add" mlir)
      assertBool "contains multiply" (T.isInfixOf "stablehlo.multiply" mlir)
      assertBool "contains compare" (T.isInfixOf "stablehlo.compare" mlir)
      assertBool "contains select" (T.isInfixOf "stablehlo.select" mlir)

  , testCase "GaussianProcessEllipticalSlice renders MLIR" $ do
      let mlir = Ex2.renderStepMlir
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains cosine" (T.isInfixOf "stablehlo.cosine" mlir)
      assertBool "contains sine" (T.isInfixOf "stablehlo.sine" mlir)
      assertBool "contains select" (T.isInfixOf "stablehlo.select" mlir)

  , testCase "LogisticRegressionHMC renders MLIR" $ do
      let mlir = Ex3.renderStepMlir
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains add" (T.isInfixOf "stablehlo.add" mlir)
      assertBool "contains multiply" (T.isInfixOf "stablehlo.multiply" mlir)
      assertBool "contains select" (T.isInfixOf "stablehlo.select" mlir)

  , testCase "BivariateGaussianMALA renders MLIR" $ do
      let mlir = Ex4.renderStepMlir
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains add" (T.isInfixOf "stablehlo.add" mlir)
      assertBool "contains multiply" (T.isInfixOf "stablehlo.multiply" mlir)
      assertBool "contains select" (T.isInfixOf "stablehlo.select" mlir)
  ]
