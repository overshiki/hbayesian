{-# LANGUAGE OverloadedStrings #-}

-- | PJRT plugin discovery and execution helpers.
--
-- This module mirrors HHLO's strategy: a 'pjrt_script.sh' downloads the
-- plugin into 'deps/pjrt/', and the Haskell code references that path by
-- default.  Users can override with the 'HBAYESIAN_PJRT_PLUGIN' environment
-- variable.
module HBayesian.HHLO.PJRT
  ( getPluginPath
  , withPJRTCPU
  , withPJRTCPUAt
  ) where

import           System.Directory    (doesFileExist)
import           System.Environment  (lookupEnv)
import           HHLO.Runtime.PJRT.Plugin (withPJRT)
import           HHLO.Runtime.PJRT.Types  (PJRTApi, PJRTClient)

-- | Return the path to the PJRT CPU plugin.
--
-- Priority:
--   1. @HBAYESIAN_PJRT_PLUGIN@ environment variable
--   2. @deps/pjrt/libpjrt_cpu.so@ (downloaded by 'scripts/pjrt_script.sh')
--   3. Runtime error with instructions
getPluginPath :: IO FilePath
getPluginPath = do
    mEnv <- lookupEnv "HBAYESIAN_PJRT_PLUGIN"
    case mEnv of
        Just p  -> return p
        Nothing -> do
            let defaultPath = "deps/pjrt/libpjrt_cpu.so"
            exists <- doesFileExist defaultPath
            if exists
                then return defaultPath
                else error $ unlines
                    [ "PJRT CPU plugin not found at: " ++ defaultPath
                    , ""
                    , "To fix this, either:"
                    , "  1. Run the download script:"
                    , "       ./scripts/pjrt_script.sh"
                    , "  2. Set the environment variable to an existing plugin:"
                    , "       export HBAYESIAN_PJRT_PLUGIN=/path/to/libpjrt_cpu.so"
                    ]

-- | Bracket-style PJRT initialization using a specific plugin path.
withPJRTCPUAt :: FilePath
              -> (HHLO.Runtime.PJRT.Types.PJRTApi -> HHLO.Runtime.PJRT.Types.PJRTClient -> IO a)
              -> IO a
withPJRTCPUAt = withPJRT

-- | Bracket-style PJRT initialization using the CPU plugin.
--
-- Equivalent to 'HHLO.Runtime.PJRT.Plugin.withPJRTCPU' but respects the
-- 'HBAYESIAN_PJRT_PLUGIN' environment variable and our 'deps/pjrt/'
-- default path.
withPJRTCPU :: (HHLO.Runtime.PJRT.Types.PJRTApi -> HHLO.Runtime.PJRT.Types.PJRTClient -> IO a) -> IO a
withPJRTCPU action = do
    path <- getPluginPath
    withPJRT path action
