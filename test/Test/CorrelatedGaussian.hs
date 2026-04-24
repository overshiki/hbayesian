{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}

module Test.CorrelatedGaussian (tests) where

import           Control.Monad       (forM_)
import           Data.List           (sort, transpose)
import           Test.Tasty
import           Test.Tasty.HUnit

import           CorrelatedGaussianHMC
import           HBayesian.Chain
import           HBayesian.Diagnostics

-----------------------------------------------------------------------------
-- Helpers
-----------------------------------------------------------------------------

-- | Sample mean of a list.
mean :: [Float] -> Float
mean xs = sum xs / fromIntegral (length xs)

-- | Sample variance (unbiased).
variance :: [Float] -> Float
variance xs =
    let m = mean xs
        n = fromIntegral (length xs)
    in if n <= 1 then 0.0
       else sum [ (x - m) ^ 2 | x <- xs ] / (n - 1)

-- | Standard deviation.
stdDev :: [Float] -> Float
stdDev xs = sqrt (variance xs)

-- | Normal CDF approximation (Abramowitz & Stegun, formula 26.2.17).
-- Accurate to ~1e-7.
normalCdf :: Float -> Float
normalCdf x =
    let z = realToFrac x :: Double
        b1 =  0.319381530
        b2 = -0.356563782
        b3 =  1.781477937
        b4 = -1.821255978
        b5 =  1.330274429
        p  =  0.2316419
        c  =  0.39894228
        t  = 1.0 / (1.0 + p * abs z)
        phi = c * exp (-0.5 * z * z)
        poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
        y = 1.0 - phi * poly
    in realToFrac $ if z > 0 then y else 1.0 - y

-- | Empirical CDF at point x for sorted samples.
empiricalCdf :: Float -> [Float] -> Float
empiricalCdf x sortedXs =
    let n = fromIntegral (length sortedXs)
        count = fromIntegral (length (takeWhile (<= x) sortedXs))
    in count / n

-- | Kolmogorov-Smirnov statistic against a theoretical CDF.
ksStatistic :: (Float -> Float) -> [Float] -> Float
ksStatistic cdf xs =
    let sorted = sort xs
        n = fromIntegral (length sorted)
        diffs = [ max (abs (empiricalCdf x sorted - cdf x))
                        (abs (empiricalCdfPrev x sorted - cdf x))
                | x <- sorted ]
        empiricalCdfPrev x sortedXs =
            let count = fromIntegral (length (takeWhile (< x) sortedXs))
            in count / n
    in maximum diffs

-- | Critical value for KS test at α = 0.01 for large n.
-- D_crit ≈ 1.628 / sqrt(n)
ksCriticalValue :: Int -> Float
ksCriticalValue n = 1.628 / sqrt (fromIntegral n)

-- | Mahalanobis distance squared: (x - μ)^T Λ (x - μ)
mahalanobisSq :: [Float] -> [Float] -> [[Float]] -> Float
mahalanobisSq x mu lam =
    let diff = zipWith (-) x mu
        -- For tridiagonal Λ, compute efficiently
        n = length diff
    in sum [ diff !! i * lam !! i !! j * diff !! j
           | i <- [0..n-1], j <- [0..n-1] ]

-- | Gelman-Rubin R-hat for a single dimension across multiple chains.
rHatSamples :: [[Float]] -> Float
rHatSamples chains =
    let m = fromIntegral (length chains)          -- number of chains
        n = fromIntegral (length (head chains))     -- iterations per chain
        chainMeans = map mean chains
        chainVars  = map variance chains
        w = mean chainVars                           -- within-chain variance
        b = n * variance chainMeans                  -- between-chain variance
        vHat = ((n - 1) / n) * w + (1 / n) * b       -- pooled variance
    in if w <= 0 then 1.0 else sqrt (vHat / w)

-----------------------------------------------------------------------------
-- Tests
-----------------------------------------------------------------------------

-- | Number of chains for R-hat test.
numChains :: Int
numChains = 4

-- | Perturbation for dispersed initial values.
perturb :: [Float] -> [Float]
perturb = map (+ 0.5)

-- | Run a single chain and return samples.
runOneChain :: IO [Float]
runOneChain = do
    (samples, _diags) <- runChainV2
    return (map head samples)  -- just first dim for some tests

tests :: TestTree
tests = testGroup "CorrelatedGaussian"
    [ testCase "HMC returns correct number of samples" testSampleCount
    , testCase "Marginal means are close to ground truth" testMarginalMeans
    , testCase "Marginal variances are close to 1.0" testMarginalVars
    , testCase "Marginal KS tests pass (dim 0)" testKS0
    , testCase "Marginal KS tests pass (dim 1)" testKS1
    , testCase "Marginal KS tests pass (dim 2)" testKS2
    , testCase "Marginal KS tests pass (dim 3)" testKS3
    , testCase "Marginal KS tests pass (dim 4)" testKS4
    , testCase "Mahalanobis distances have correct mean" testMahalanobisMean
    , testCase "R-hat < 1.1 across 4 chains" testRhat
    ]

-- | Test that we get the expected number of samples.
testSampleCount :: Assertion
testSampleCount = do
    (samples, _diags) <- runChainV2
    length samples @?= 2000

-- | Test that each marginal mean is within 3 SE of the true mean.
-- For AR(1) with uniform variance, marginal SD = 1, so SE = 1/sqrt(2000) ≈ 0.022.
testMarginalMeans :: Assertion
testMarginalMeans = do
    (samples, _diags) <- runChainV2
    let n = fromIntegral (length samples) :: Float
        se = 1.0 / sqrt n   -- marginal SD = 1 for all dims
        thresh = 3.0 * se   -- ~0.067
    forM_ (zip [0..4] targetMean) $ \(i, muTrue) -> do
        let xs = map (!! i) samples
            muHat = mean xs
            err = abs (muHat - muTrue)
        assertBool ("dim " ++ show i ++ " mean off by " ++ show err)
                   (err < thresh)

-- | Test that each marginal variance is close to 1.0.
testMarginalVars :: Assertion
testMarginalVars = do
    (samples, _diags) <- runChainV2
    let thresh = 0.15   -- generous for 2000 samples
    forM_ [0..4] $ \i -> do
        let xs = map (!! i) samples
            varHat = variance xs
            err = abs (varHat - 1.0)
        assertBool ("dim " ++ show i ++ " variance off by " ++ show err)
                   (err < thresh)

-- | Marginal KS tests for each dimension.
-- The marginal of an AR(1) Gaussian is N(mu_i, 1).
testKS :: Int -> Assertion
testKS i = do
    (samples, _diags) <- runChainV2
    let xs = map (!! i) samples
        mu = targetMean !! i
        cdf x = normalCdf ((x - mu) / 1.0)
        stat = ksStatistic cdf xs
        crit = ksCriticalValue (length xs)
    assertBool ("dim " ++ show i ++ " KS stat " ++ show stat ++ " > crit " ++ show crit)
               (stat < crit)

testKS0, testKS1, testKS2, testKS3, testKS4 :: Assertion
testKS0 = testKS 0
testKS1 = testKS 1
testKS2 = testKS 2
testKS3 = testKS 3
testKS4 = testKS 4

-- | Test that mean of Mahalanobis distances ≈ k = 5.
-- E[d²] = k for χ²_k.
testMahalanobisMean :: Assertion
testMahalanobisMean = do
    (samples, _diags) <- runChainV2
    let dists = [ mahalanobisSq s targetMean targetPrecision | s <- samples ]
        meanD = mean dists
        -- For χ²_5, E[d²] = 5, Var[d²] = 10
        -- SE of mean = sqrt(10/n) ≈ 0.071
        se = sqrt (10.0 / fromIntegral (length samples))
        thresh = 3.0 * se   -- ~0.21
        err = abs (meanD - 5.0)
    assertBool ("Mahalanobis mean " ++ show meanD ++ " off by " ++ show err)
               (err < thresh)

-- | Test Gelman-Rubin across 4 chains.
testRhat :: Assertion
testRhat = do
    let config = defaultChainConfig { ccNumIterations = 500 }
        ck = compileHMC (makeKernel (HMCConfig 0.1 10))
                        gaussianLogPdf gaussianGrad
    results <- parallelChains numChains perturb ck
                    (replicate targetDim 0.0) config
    let chains = map fst results
        -- Transpose: get per-dimension chains
        perDimChains = [ [ map (!! i) c | c <- chains ] | i <- [0..4] ]
        rhats = map rHatSamples perDimChains
    forM_ (zip [0..4] rhats) $ \(i, r) -> do
        assertBool ("dim " ++ show i ++ " R-hat = " ++ show r ++ " >= 1.1")
                   (r < 1.1)
