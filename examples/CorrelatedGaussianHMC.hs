{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example: HMC on a 5-D correlated Gaussian with AR(1) covariance.
--
-- This example serves as a rigorous regression test for the HMC
-- implementation. The target distribution is known analytically, so
-- we can apply statistical goodness-of-fit tests to the samples.
--
-- Target:  θ ~ N(μ, Σ)  where Σ_ij = ρ^|i-j|  (AR(1), uniform variance)
--
-- The precision matrix Λ = Σ⁻¹ is tridiagonal, making both log-density
-- and gradient easy to write in Builder.
module CorrelatedGaussianHMC
  ( -- * Ground truth
    targetMean
  , targetRho
  , targetDim
  , targetCov
  , targetPrecision
    -- * Model
  , gaussianLogPdf
  , gaussianGrad
  , makeKernel
    -- * HMC config (re-exported for test use)
  , HMCConfig (..)
    -- * Execution
  , runChain
  , runChainV2
    -- * Goodness-of-fit
  , goodnessOfFitReport
    -- * Debug
  , renderStepMlir
  ) where

import           Data.List           (sort)
import           Data.Text           (Text)
import           Data.Word           (Word64)
import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops hiding (map, maximum, sort, sqrt)
import           HBayesian.HHLO.PJRT
import           HBayesian.MCMC.HMC
import           HBayesian.Chain
import           Common

-----------------------------------------------------------------------------
-- Ground truth
-----------------------------------------------------------------------------

-- | Dimensionality.
targetDim :: Int
targetDim = 5

-- | True mean vector μ.
targetMean :: [Float]
targetMean = [1.0, -0.5, 2.0, 0.0, -1.0]

-- | AR(1) correlation coefficient.
targetRho :: Float
targetRho = 0.7

-- | Convenience: ρ².
rho2 :: Float
rho2 = targetRho * targetRho

-- | Precision matrix entries (analytical form for AR(1) with uniform variance).
-- Λ_ii for boundaries (i = 0, 4).
precBoundary :: Float
precBoundary = 1.0 / (1.0 - rho2)

-- | Λ_ii for interior points (i = 1, 2, 3).
precInterior :: Float
precInterior = (1.0 + rho2) / (1.0 - rho2)

-- | Λ_i,i+1 = Λ_i+1,i (off-diagonal).
precOffDiag :: Float
precOffDiag = -targetRho / (1.0 - rho2)

-- | Covariance matrix Σ_ij = ρ^|i-j| (for reference / validation).
targetCov :: [[Float]]
targetCov =
  [ [ targetRho ^ abs (i - j) | j <- [0 .. targetDim - 1] ]
  | i <- [0 .. targetDim - 1] ]

-- | Precision matrix Λ = Σ⁻¹ (tridiagonal, analytical form).
targetPrecision :: [[Float]]
targetPrecision =
  [ [ case abs (i - j) of
        0 | i == 0 || i == targetDim - 1 -> precBoundary
          | otherwise                    -> precInterior
        1 -> precOffDiag
        _ -> 0.0
    | j <- [0 .. targetDim - 1] ]
  | i <- [0 .. targetDim - 1] ]

-----------------------------------------------------------------------------
-- Model: log-density
-----------------------------------------------------------------------------

-- | Compute the quadratic form (θ - μ)^T Λ (θ - μ) for tridiagonal Λ.
--
-- Since Λ is symmetric tridiagonal, this expands to:
--   Σ_i Λ_ii * d_i² + 2 * Σ_i Λ_i,i+1 * d_i * d_{i+1}
quadraticForm :: Tensor '[5] 'F32 -> Builder (Tensor '[] 'F32)
quadraticForm theta = do
    -- diffs d_i = theta_i - mu_i
    d0 <- diff theta 0
    d1 <- diff theta 1
    d2 <- diff theta 2
    d3 <- diff theta 3
    d4 <- diff theta 4

    -- Diagonal terms: Λ_ii * d_i²
    q0 <- diagTerm precBoundary d0
    q1 <- diagTerm precInterior d1
    q2 <- diagTerm precInterior d2
    q3 <- diagTerm precInterior d3
    q4 <- diagTerm precBoundary d4

    -- Off-diagonal terms: 2 * Λ_i,i+1 * d_i * d_{i+1}
    q01 <- offDiagTerm precOffDiag d0 d1
    q12 <- offDiagTerm precOffDiag d1 d2
    q23 <- offDiagTerm precOffDiag d2 d3
    q34 <- offDiagTerm precOffDiag d3 d4

    -- Sum everything (all terms are scalars)
    s1 <- tadd q0 q01
    s2 <- tadd s1 q1
    s3 <- tadd s2 q12
    s4 <- tadd s3 q2
    s5 <- tadd s4 q23
    s6 <- tadd s5 q3
    s7 <- tadd s6 q34
    tadd s7 q4
  where
    diff th idx = do
        thi <- tslice1 @5 @'F32 th (fromIntegral idx)
        mui <- tconstant (realToFrac (targetMean !! idx))
        tsub thi mui
    diagTerm lam di = do
        lamT <- tconstant (realToFrac lam)
        di2 <- tmul di di
        tmul lamT di2
    offDiagTerm lam di dj = do
        lamT <- tconstant (realToFrac lam)
        two  <- tconstant 2.0
        prod <- tmul di dj
        tmp <- tmul lamT prod
        tmul two tmp

gaussianLogPdf :: Tensor '[5] 'F32 -> Builder (Tensor '[] 'F32)
gaussianLogPdf theta = do
    quad <- quadraticForm theta
    negHalf <- tconstant (-0.5)
    tmul negHalf quad

-----------------------------------------------------------------------------
-- Gradient
-----------------------------------------------------------------------------

-- | ∇_θ log p(θ) = -Λ (θ - μ)
--
-- For tridiagonal Λ:
--   g_0 = -(Λ_00*d0 + Λ_01*d1)
--   g_i = -(Λ_{i,i-1}*d_{i-1} + Λ_ii*d_i + Λ_{i,i+1}*d_{i+1})
--   g_4 = -(Λ_43*d3 + Λ_44*d4)
gaussianGrad :: Gradient '[5] 'F32
gaussianGrad theta = do
    d0 <- diff theta 0
    d1 <- diff theta 1
    d2 <- diff theta 2
    d3 <- diff theta 3
    d4 <- diff theta 4

    g0 <- grad0 d0 d1
    g1 <- grad1 d0 d1 d2
    g2 <- grad2 d1 d2 d3
    g3 <- grad3 d2 d3 d4
    g4 <- grad4 d3 d4

    g0r <- treshape @'[] @'[1] g0
    g1r <- treshape @'[] @'[1] g1
    g2r <- treshape @'[] @'[1] g2
    g3r <- treshape @'[] @'[1] g3
    g4r <- treshape @'[] @'[1] g4
    concatenate @'[1] @'[5] @'F32 0 [g0r, g1r, g2r, g3r, g4r]
  where
    diff th idx = do
        thi <- tslice1 @5 @'F32 th (fromIntegral idx)
        mui <- tconstant (realToFrac (targetMean !! idx))
        tsub thi mui

    grad0 d0 d1 = do
        lam0 <- tconstant (realToFrac precBoundary)
        lam1 <- tconstant (realToFrac precOffDiag)
        t0 <- tmul lam0 d0
        t1 <- tmul lam1 d1
        s  <- tadd t0 t1
        tnegate s

    grad1 d0 d1 d2 = do
        lam0 <- tconstant (realToFrac precOffDiag)
        lam1 <- tconstant (realToFrac precInterior)
        lam2 <- tconstant (realToFrac precOffDiag)
        t0 <- tmul lam0 d0
        t1 <- tmul lam1 d1
        t2 <- tmul lam2 d2
        s1 <- tadd t0 t1
        s  <- tadd s1 t2
        tnegate s

    grad2 d1 d2 d3 = do
        lam1 <- tconstant (realToFrac precOffDiag)
        lam2 <- tconstant (realToFrac precInterior)
        lam3 <- tconstant (realToFrac precOffDiag)
        t1 <- tmul lam1 d1
        t2 <- tmul lam2 d2
        t3 <- tmul lam3 d3
        s1 <- tadd t1 t2
        s  <- tadd s1 t3
        tnegate s

    grad3 d2 d3 d4 = do
        lam2 <- tconstant (realToFrac precOffDiag)
        lam3 <- tconstant (realToFrac precInterior)
        lam4 <- tconstant (realToFrac precOffDiag)
        t2 <- tmul lam2 d2
        t3 <- tmul lam3 d3
        t4 <- tmul lam4 d4
        s1 <- tadd t2 t3
        s  <- tadd s1 t4
        tnegate s

    grad4 d3 d4 = do
        lam3 <- tconstant (realToFrac precOffDiag)
        lam4 <- tconstant (realToFrac precBoundary)
        t3 <- tmul lam3 d3
        t4 <- tmul lam4 d4
        s  <- tadd t3 t4
        tnegate s

-----------------------------------------------------------------------------
-- Kernel
-----------------------------------------------------------------------------

makeKernel :: HMCConfig -> Kernel '[5] 'F32 (HMCState '[5] 'F32) (Info '[5] 'F32)
makeKernel config = hmc gaussianLogPdf gaussianGrad config

-----------------------------------------------------------------------------
-- Tier A: render MLIR
-----------------------------------------------------------------------------

renderStepMlir :: Text
renderStepMlir =
  renderKernelStep @'[5] @'F32
    [ FuncArg "key" (TensorType [2] UI64)
    , FuncArg "pos" (TensorType [5] F32)
    , FuncArg "p"   (TensorType [5] F32)
    , FuncArg "ld"  (TensorType [] F32)
    , FuncArg "g"   (TensorType [5] F32)
    ] $ do
      key <- arg @'[2] @'UI64
      pos <- arg @'[5] @'F32
      p   <- arg @'[5] @'F32
      ld  <- arg @'[] @'F32
      g   <- arg @'[5] @'F32
      let config = HMCConfig { hmcStepSize = 0.1, hmcNumLeapfrogSteps = 10 }
      (state', _info) <- kernelStep (makeKernel config) (Key key) (HMCState pos p ld g)
      return (hmcPosition state')

-----------------------------------------------------------------------------
-- Tier B: run chain (v0.1 style)
-----------------------------------------------------------------------------

runChain :: IO [[Float]]
runChain = withPJRTCPU $ \api client -> do
    let config = HMCConfig { hmcStepSize = 0.1, hmcNumLeapfrogSteps = 10 }
        kernel = makeKernel config

    let ldMod = moduleFromBuilder @'[] @'F32 "main"
                  [ FuncArg "theta" (TensorType [5] F32) ] $ do
          theta <- arg @'[5] @'F32
          gaussianLogPdf theta
    ldExe <- compileModule api client ldMod

    let gradMod = moduleFromBuilder @'[5] @'F32 "main"
                    [ FuncArg "theta" (TensorType [5] F32) ] $ do
          theta <- arg @'[5] @'F32
          gaussianGrad theta
    gradExe <- compileModule api client gradMod

    let stepMod = moduleFromBuilder @'[5] @'F32 "main"
                    [ FuncArg "key" (TensorType [2] UI64)
                    , FuncArg "pos" (TensorType [5] F32)
                    , FuncArg "p"   (TensorType [5] F32)
                    , FuncArg "ld"  (TensorType [] F32)
                    , FuncArg "g"   (TensorType [5] F32)
                    ] $ do
          key <- arg @'[2] @'UI64
          pos <- arg @'[5] @'F32
          p   <- arg @'[5] @'F32
          ld  <- arg @'[] @'F32
          g   <- arg @'[5] @'F32
          (state', _info) <- kernelStep kernel (Key key) (HMCState pos p ld g)
          return (hmcPosition state')
    stepExe <- compileModule api client stepMod

    let seed :: Word64 = 42
        pos0 = replicate targetDim 0.0

    posBuf0 <- bufferFromF32 api client [targetDim] pos0
    [ldBuf0] <- executeModule api ldExe [posBuf0]
    [ld0] <- bufferToF32 api ldBuf0 1
    [gBuf0] <- executeModule api gradExe [posBuf0]
    g0 <- bufferToF32 api gBuf0 targetDim

    loop api client stepExe ldExe gradExe seed (0 :: Int) pos0 ld0 g0 (10 :: Int) []
  where
    loop _ _ _ _ _ _ _ _ _ _ 0 acc = return (reverse acc)
    loop api client stepExe ldExe gradExe seed step pos ld g n acc = do
        let key = [seed, fromIntegral step]
            zeroP = replicate targetDim 0.0
        keyBuf <- bufferFromUI64 api client [2] key
        posBuf <- bufferFromF32  api client [targetDim] pos
        pBuf   <- bufferFromF32  api client [targetDim] zeroP
        ldBuf  <- bufferFromF32  api client [] [ld]
        gBuf   <- bufferFromF32  api client [targetDim] g
        [newPosBuf] <- executeModule api stepExe [keyBuf, posBuf, pBuf, ldBuf, gBuf]
        newPos <- bufferToF32 api newPosBuf targetDim
        [newLdBuf] <- executeModule api ldExe [newPosBuf]
        [newLd] <- bufferToF32 api newLdBuf 1
        [newGBuf] <- executeModule api gradExe [newPosBuf]
        newG <- bufferToF32 api newGBuf targetDim
        loop api client stepExe ldExe gradExe seed (step + 1) newPos newLd newG (n - 1) (newPos : acc)

-----------------------------------------------------------------------------
-- Tier B: run chain (v0.2 style)
-----------------------------------------------------------------------------

runChainV2 :: IO ([[Float]], [Diagnostic])
runChainV2 = do
    let config = HMCConfig { hmcStepSize = 0.1, hmcNumLeapfrogSteps = 10 }
        kernel = makeKernel config
        ck     = compileHMC kernel gaussianLogPdf gaussianGrad
    sampleChain ck (replicate targetDim 0.0) $
        burnIn 500 $ thin 2 $ defaultChainConfig
            { ccNumIterations = 2000
            , ccSeed = 42
            }

-----------------------------------------------------------------------------
-- Goodness-of-fit report
-----------------------------------------------------------------------------

meanF :: [Float] -> Float
meanF xs = sum xs / fromIntegral (length xs)

varianceF :: [Float] -> Float
varianceF xs =
    let m = meanF xs
        n = fromIntegral (length xs)
    in if n <= 1 then 0.0
       else sum [ (x - m) * (x - m) | x <- xs ] / (n - 1)

-- | Normal CDF approximation (Abramowitz & Stegun, formula 26.2.17).
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

empiricalCdf :: Float -> [Float] -> Float
empiricalCdf x sortedXs =
    let n = fromIntegral (length sortedXs)
        count = fromIntegral (length (takeWhile (<= x) sortedXs))
    in count / n

ksStatistic :: (Float -> Float) -> [Float] -> Float
ksStatistic cdf xs =
    let sorted = sort xs
        n = fromIntegral (length sorted)
        empiricalCdfPrev x sortedXs =
            let count = fromIntegral (length (takeWhile (< x) sortedXs))
            in count / n
        diffs = [ max (abs (empiricalCdf x sorted - cdf x))
                        (abs (empiricalCdfPrev x sorted - cdf x))
                | x <- sorted ]
    in maximum diffs

ksCriticalValue :: Int -> Float
ksCriticalValue n = 1.628 / sqrt (fromIntegral n)

mahalanobisSq :: [Float] -> [Float] -> [[Float]] -> Float
mahalanobisSq x mu lam =
    let diff = zipWith (-) x mu
        n = length diff
    in sum [ diff !! i * lam !! i !! j * diff !! j
           | i <- [0..n-1], j <- [0..n-1] ]

-- | Print a formatted goodness-of-fit report for HMC samples.
goodnessOfFitReport :: [[Float]] -> IO ()
goodnessOfFitReport samples = do
    let n = length samples
    putStrLn "=================================="
    putStrLn "  Goodness-of-Fit Report"
    putStrLn "=================================="
    putStrLn $ "Sample count: " ++ show n
    putStrLn ""

    -- Marginal means
    putStrLn "Marginal means (expected vs observed):"
    forM_ (zip [0..4] targetMean) $ \(i, muTrue) -> do
        let xs = map (!! i) samples
            muHat = meanF xs
            err = abs (muHat - muTrue)
            se = 1.0 / sqrt (fromIntegral n)
            status = if err < 3.0 * se then "PASS" else "FAIL"
        putStrLn $ "  dim " ++ show i ++ ":  " ++ padL 6 (show muTrue)
                 ++ " vs  " ++ padL 6 (show muHat)
                 ++ "  (diff: " ++ padL 6 (show err) ++ ")  " ++ status

    putStrLn ""
    putStrLn "Marginal variances (expected vs observed):"
    forM_ [0..4] $ \i -> do
        let xs = map (!! i) samples
            varHat = varianceF xs
            err = abs (varHat - 1.0)
            status = if err < 0.15 then "PASS" else "FAIL"
        putStrLn $ "  dim " ++ show i ++ ":  " ++ padL 6 ("1.0")
                 ++ " vs  " ++ padL 6 (show varHat)
                 ++ "  (diff: " ++ padL 6 (show err) ++ ")  " ++ status

    putStrLn ""
    let crit = ksCriticalValue n
    putStrLn $ "KS tests (critical value: " ++ show crit ++ "):"
    forM_ [0..4] $ \i -> do
        let xs = map (!! i) samples
            mu = targetMean !! i
            cdf x = normalCdf ((x - mu) / 1.0)
            stat = ksStatistic cdf xs
            status = if stat < crit then "PASS" else "FAIL"
        putStrLn $ "  dim " ++ show i ++ ":  stat=" ++ padL 8 (show stat)
                 ++ "  " ++ status

    putStrLn ""
    let dists = [ mahalanobisSq s targetMean targetPrecision | s <- samples ]
        meanD = meanF dists
        se = sqrt (10.0 / fromIntegral n)
        thresh = 3.0 * se
        err = abs (meanD - 5.0)
        status = if err < thresh then "PASS" else "FAIL"
    putStrLn "Mahalanobis distances:"
    putStrLn $ "  expected mean: 5.00"
    putStrLn $ "  observed mean: " ++ show meanD ++ "  " ++ status
    putStrLn ""
  where
    padL w s = replicate (max 0 (w - length s)) ' ' ++ s
    forM_ = flip mapM_
