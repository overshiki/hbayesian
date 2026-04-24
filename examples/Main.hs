module Main (main) where

import           System.Environment  (getArgs, setEnv)
import           System.Exit         (exitFailure)
import qualified Data.Text.IO as T
import qualified LinearRegressionRandomWalk as Ex1
import qualified GaussianProcessEllipticalSlice as Ex2
import qualified LogisticRegressionHMC as Ex3
import qualified BivariateGaussianMALA as Ex4

-----------------------------------------------------------------------------
-- CLI
-----------------------------------------------------------------------------

data Mode = ModeRender | ModeExecute
    deriving (Eq, Show)

data Options = Options
    { optMode       :: !Mode
    , optPJRTPlugin :: !(Maybe FilePath)
    }

defaultOptions :: Options
defaultOptions = Options
    { optMode       = ModeRender
    , optPJRTPlugin = Nothing
    }

usage :: String
usage = unlines
    [ "Usage: hbayesian-examples [OPTIONS]"
    , ""
    , "Options:"
    , "  --render               Print StableHLO MLIR for all examples (default)"
    , "  --execute              Run MCMC chains on PJRT"
    , "  --pjrt-plugin PATH     Use a custom PJRT plugin (default: deps/pjrt/libpjrt_cpu.so)"
    , "  --help                 Show this message"
    ]

parseArgs :: [String] -> Either String Options
parseArgs = go defaultOptions
  where
    go opts [] = Right opts
    go opts ("--render"       : rest) = go (opts { optMode = ModeRender }) rest
    go opts ("--execute"      : rest) = go (opts { optMode = ModeExecute }) rest
    go opts ("--pjrt-plugin"  : path : rest) = go (opts { optPJRTPlugin = Just path }) rest
    go _    ("--help"         : _) = Left usage
    go _   (bad               : _) = Left $ "Unknown flag: " ++ bad ++ "\n" ++ usage

-----------------------------------------------------------------------------
-- Main
-----------------------------------------------------------------------------

main :: IO ()
main = do
    args <- getArgs
    opts <- case parseArgs args of
        Left msg -> putStrLn msg >> exitFailure
        Right o  -> return o

    case optMode opts of
        ModeRender  -> runRender
        ModeExecute -> runExecute (optPJRTPlugin opts)

runRender :: IO ()
runRender = do
    putStrLn "=== Example 1: Bayesian Linear Regression (RandomWalk) ==="
    T.putStrLn Ex1.renderStepMlir

    putStrLn "\n=== Example 2: Gaussian Process (EllipticalSlice) ==="
    T.putStrLn Ex2.renderStepMlir

    putStrLn "\n=== Example 3: Logistic Regression (HMC) ==="
    T.putStrLn Ex3.renderStepMlir

    putStrLn "\n=== Example 4: Bivariate Gaussian (MALA) ==="
    T.putStrLn Ex4.renderStepMlir

runExecute :: Maybe FilePath -> IO ()
runExecute mPluginPath = do
    -- If a custom plugin path is given, expose it via the env var so that
    -- the example modules' withPJRTCPU picks it up.
    case mPluginPath of
        Just path -> setEnv "HBAYESIAN_PJRT_PLUGIN" path
        Nothing   -> return ()

    putStrLn "=== Example 1: Bayesian Linear Regression (RandomWalk) ==="
    samples1 <- Ex1.runChain
    mapM_ print samples1

    putStrLn "\n=== Example 2: Gaussian Process (EllipticalSlice) ==="
    samples2 <- Ex2.runChain
    mapM_ print samples2

    putStrLn "\n=== Example 3: Logistic Regression (HMC) ==="
    samples3 <- Ex3.runChain
    mapM_ print samples3

    putStrLn "\n=== Example 4: Bivariate Gaussian (MALA) ==="
    samples4 <- Ex4.runChain
    mapM_ print samples4
