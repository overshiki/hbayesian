{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}

module Test.CorrelatedGaussianNUTS (tests) where

import           Test.Tasty
import           Test.Tasty.HUnit

import           CorrelatedGaussianHMC (targetMean, targetDim, gaussianLogPdf, gaussianGrad)
import           CorrelatedGaussianNUTS (runChainV2)
import           HBayesian.Chain
import           HBayesian.Diagnostics

mean :: [Float] -> Float
mean xs = sum xs / fromIntegral (length xs)

tests :: TestTree
tests = testGroup "CorrelatedGaussianNUTS"
    [ testCase "NUTS returns correct number of samples" testSampleCount
    , testCase "NUTS marginal mean (dim 0) is close to ground truth" testMarginalMean0
    ]

testSampleCount :: Assertion
testSampleCount = do
    (samples, _diags) <- runChainV2
    length samples @?= 1000

testMarginalMean0 :: Assertion
testMarginalMean0 = do
    (samples, _diags) <- runChainV2
    let n = fromIntegral (length samples) :: Float
        se = 1.0 / sqrt n
        thresh = 3.0 * se
        xs = map head samples
        muHat = mean xs
        err = abs (muHat - head targetMean)
    assertBool ("dim 0 mean off by " ++ show err)
               (err < thresh)
