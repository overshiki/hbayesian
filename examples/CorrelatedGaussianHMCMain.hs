{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import HBayesian.Diagnostics (acceptanceRate)
import qualified CorrelatedGaussianHMC as Ex

main :: IO ()
main = do
    putStrLn "Running HMC on 5-D correlated Gaussian (AR(1), rho=0.7)..."
    putStrLn ""
    (samples, diags) <- Ex.runChainV2
    Ex.goodnessOfFitReport samples
    putStrLn $ "Acceptance rate: " ++ show (acceptanceRate diags)
