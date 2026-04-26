module Main (main) where

import CorrelatedGaussianNUTS (runChainV2)

main :: IO ()
main = do
    putStrLn "Running NUTS on 5-D correlated Gaussian..."
    (samples, _diags) <- runChainV2
    putStrLn $ "Generated " ++ show (length samples) ++ " samples"
    putStrLn $ "First sample: " ++ show (head samples)
    putStrLn $ "Last sample:  " ++ show (last samples)
