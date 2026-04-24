{-# LANGUAGE ScopedTypeVariables #-}

-- | Diagnostics for MCMC chains.
--
-- These functions operate on host-side 'Diagnostic' records collected
-- by 'HBayesian.Chain.sampleChain'.
module HBayesian.Diagnostics
  ( -- * Acceptance diagnostics
    acceptanceRate
  , meanAcceptProb
    -- * Convergence diagnostics
  , rHat
    -- * Effective sample size
  , ess
  ) where

import           HBayesian.Chain     (Diagnostic (..))

-----------------------------------------------------------------------------
-- Acceptance diagnostics
-----------------------------------------------------------------------------

-- | Fraction of steps that were accepted.
acceptanceRate :: [Diagnostic] -> Double
acceptanceRate diags =
    let accepted = length (filter dAccepted diags)
        total    = length diags
    in if total == 0 then 0.0 else fromIntegral accepted / fromIntegral total

-- | Mean acceptance probability across all steps.
meanAcceptProb :: [Diagnostic] -> Double
meanAcceptProb diags =
    let total = length diags
    in if total == 0 then 0.0
       else sum (map (realToFrac . dAcceptProb) diags) / fromIntegral total

-----------------------------------------------------------------------------
-- Convergence diagnostics
-----------------------------------------------------------------------------

-- | Potential scale reduction factor (R-hat).
--
-- Requires at least two chains. Returns 'Nothing' if insufficient data.
rHat :: [[Diagnostic]] -> Maybe Double
rHat chains
    | length chains < 2 = Nothing
    | any null chains   = Nothing
    | otherwise         = Just $ computeRHat (map (map dAcceptProb) chains)

computeRHat :: [[Float]] -> Double
computeRHat chains =
    let n = fromIntegral (length (head chains)) :: Double
        -- Within-chain variance
        chainVars = map varianceF chains
        w = meanD chainVars
        -- Between-chain variance
        b = n * varianceD (map meanF chains)
        -- Pooled variance
        vHat = ((n - 1) / n) * w + (1 / n) * b
    in sqrt (vHat / w)

meanF :: [Float] -> Double
meanF xs = sum (map realToFrac xs) / fromIntegral (length xs)

meanD :: [Double] -> Double
meanD xs = sum xs / fromIntegral (length xs)

varianceF :: [Float] -> Double
varianceF xs =
    let m = meanF xs
        n = fromIntegral (length xs) :: Double
    in if n <= 1 then 0.0
       else sum (map (\x -> (realToFrac x - m) ** 2) xs) / (n - 1)

varianceD :: [Double] -> Double
varianceD xs =
    let m = meanD xs
        n = fromIntegral (length xs) :: Double
    in if n <= 1 then 0.0
       else sum (map (\x -> (x - m) ** 2) xs) / (n - 1)

-----------------------------------------------------------------------------
-- Effective sample size
-----------------------------------------------------------------------------

-- | Effective sample size estimate (naive autocorrelation method).
--
-- This is a simplified ESS that uses the autocorrelation of the
-- acceptance probabilities as a proxy for chain mixing.
ess :: [Diagnostic] -> Double
ess diags
    | null diags = 0.0
    | otherwise  =
        let n    = fromIntegral (length diags)
            vals = map (realToFrac . dAcceptProb) diags
            rhoK = takeWhile (> 0.05) (autocorrelations vals)
        in n / (1 + 2 * sum rhoK)

-- | Autocorrelation at lag k.
autocorrelations :: [Double] -> [Double]
autocorrelations xs =
    let n = length xs
        m = meanD xs
        v = varianceD xs
        centred = map (\x -> x - m) xs
    in if v == 0 then []
       else [ lagKAutocorr centred k / v | k <- [1 .. n `div` 2] ]

lagKAutocorr :: [Double] -> Int -> Double
lagKAutocorr xs k =
    let n = length xs
    in if k >= n then 0.0
       else sum [ xs !! i * xs !! (i + k) | i <- [0 .. n - k - 1] ] / fromIntegral (n - k)
