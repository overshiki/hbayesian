{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}

module Test.NealFunnel (tests) where

import           Control.Monad       (forM_)
import           Data.List           (transpose)
import           Test.Tasty
import           Test.Tasty.HUnit

import           NealFunnel
import           HBayesian.Chain

-----------------------------------------------------------------------------
-- ESS computation on raw samples
-----------------------------------------------------------------------------

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

variance :: [Double] -> Double
variance xs =
    let m = mean xs
        n = fromIntegral (length xs)
    in if n <= 1 then 0.0
       else sum [ (x - m) ^ 2 | x <- xs ] / (n - 1)

-- | Autocorrelation at lag k.
autocorr :: [Double] -> Int -> Double
autocorr xs k =
    let n = length xs
        m = mean xs
        v = variance xs
        c = map (\x -> x - m) xs
    in if v == 0 || k >= n then 0.0
       else sum [ c !! i * c !! (i + k) | i <- [0 .. n - k - 1] ] / fromIntegral (n - k) / v

-- | Effective sample size for a 1-D chain.
ess1D :: [Double] -> Double
ess1D xs =
    let n = fromIntegral (length xs)
        rhos = takeWhile (> 0.05) [ autocorr xs k | k <- [1 .. length xs `div` 2] ]
    in n / (1 + 2 * sum rhos)

-- | ESS per dimension, returning (min_ess, mean_ess, per_dim_ess).
essChain :: [[Float]] -> (Double, Double, [Double])
essChain samples =
    let perDim = map (map realToFrac) (transpose samples)
        essVals = map ess1D perDim
        minEss = minimum essVals
        meanEss = mean essVals
    in (minEss, meanEss, essVals)

-----------------------------------------------------------------------------
-- Run experiment
-----------------------------------------------------------------------------

-- | Run a single configuration and return (ess_min, ess_mean).
runConfig :: String -> IO ([[Float]], [Diagnostic]) -> IO (Double, Double)
runConfig label action = do
    (samples, _diags) <- action
    let (minE, meanE, _) = essChain samples
    return (minE, meanE)

-- | Run all 4 configurations and report.
runExperiment :: IO [(String, Double, Double)]
runExperiment = do
    putStrLn "\n=== Neal's Funnel Experiment ==="
    putStrLn "Running HMC (L=10)..."
    (min10, mean10) <- runConfig "HMC-10" (runHMC 10)
    putStrLn $ "  min ESS=" ++ show min10 ++ " mean ESS=" ++ show mean10

    putStrLn "Running HMC (L=50)..."
    (min50, mean50) <- runConfig "HMC-50" (runHMC 50)
    putStrLn $ "  min ESS=" ++ show min50 ++ " mean ESS=" ++ show mean50

    putStrLn "Running HMC (L=200)..."
    (min200, mean200) <- runConfig "HMC-200" (runHMC 200)
    putStrLn $ "  min ESS=" ++ show min200 ++ " mean ESS=" ++ show mean200

    putStrLn "Running NUTS..."
    (minN, meanN) <- runConfig "NUTS" runNUTS
    putStrLn $ "  min ESS=" ++ show minN ++ " mean ESS=" ++ show meanN

    return
        [ ("HMC-L10", min10, mean10)
        , ("HMC-L50", min50, mean50)
        , ("HMC-L200", min200, mean200)
        , ("NUTS", minN, meanN)
        ]

-----------------------------------------------------------------------------
-- Tests
-----------------------------------------------------------------------------

tests :: TestTree
tests = testGroup "NealFunnel"
    [ testCase "NUTS mean-ESS exceeds all fixed-step HMC configs" testNUTSMeanESS
    , testCase "NUTS mean-ESS is at least 3x HMC-200 mean-ESS" testNUTSEfficiency
    ]

testNUTSMeanESS :: Assertion
testNUTSMeanESS = do
    results <- runExperiment
    let (_, minN, meanN) = head (filter (\(n,_,_) -> n == "NUTS") results)
        hmcResults = filter (\(n,_,_) -> n /= "NUTS") results
    forM_ hmcResults $ \(name, _minE, meanE) -> do
        assertBool ("NUTS mean-ESS " ++ show meanN ++ " should exceed " ++ name ++ " mean-ESS " ++ show meanE)
                   (meanN > meanE)
    -- Also check that NUTS min-ESS is reasonable (not catastrophically low)
    assertBool ("NUTS min-ESS " ++ show minN ++ " should be > 5")
               (minN > 5.0)

testNUTSEfficiency :: Assertion
testNUTSEfficiency = do
    results <- runExperiment
    let (_, _minN, meanN) = head (filter (\(n,_,_) -> n == "NUTS") results)
        (_, _, mean200) = head (filter (\(n,_,_) -> n == "HMC-L200") results)
        ratio = meanN / mean200
    assertBool ("NUTS mean-ESS / HMC-200 mean-ESS = " ++ show ratio ++ " should be >= 3.0")
               (ratio >= 3.0)
