module Main (main) where

import qualified Data.Text.IO as T
import qualified LinearRegressionRandomWalk as Ex1
import qualified GaussianProcessEllipticalSlice as Ex2
import qualified LogisticRegressionHMC as Ex3
import qualified BivariateGaussianMALA as Ex4

main :: IO ()
main = do
  putStrLn "=== Example 1: Bayesian Linear Regression (RandomWalk) ==="
  T.putStrLn Ex1.renderStepMlir

  putStrLn "\n=== Example 2: Gaussian Process (EllipticalSlice) ==="
  T.putStrLn Ex2.renderStepMlir

  putStrLn "\n=== Example 3: Logistic Regression (HMC) ==="
  T.putStrLn Ex3.renderStepMlir

  putStrLn "\n=== Example 4: Bivariate Gaussian (MALA) ==="
  T.putStrLn Ex4.renderStepMlir
