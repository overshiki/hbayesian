{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}

module Test.Chain (tests) where

import           Test.Tasty
import           Test.Tasty.HUnit

import qualified LinearRegressionRandomWalk as Ex1
import qualified BivariateGaussianMALA as Ex4

import           HBayesian.Chain
import           HBayesian.Diagnostics

tests :: TestTree
tests = testGroup "Chain"
    [ testCase "LinearRegressionRandomWalk V2 runs and returns 10 samples" $ do
        (samples, diags) <- Ex1.runChainV2
        length samples @?= 10
        length diags @?= 10

    , testCase "LinearRegressionRandomWalk V2 acceptance rate is in [0,1]" $ do
        (_samples, diags) <- Ex1.runChainV2
        let rate = acceptanceRate diags
        assertBool "acceptance rate should be >= 0" (rate >= 0)
        assertBool "acceptance rate should be <= 1" (rate <= 1)

    , testCase "LinearRegressionRandomWalk V2 mean accept prob is in [0,1]" $ do
        (_samples, diags) <- Ex1.runChainV2
        let m = meanAcceptProb diags
        assertBool "mean accept prob should be >= 0" (m >= 0)
        assertBool "mean accept prob should be <= 1" (m <= 1)
    ]
