{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

-- | Chain combinators for HBayesian v0.2.
--
-- This module hides the mechanical work of PJRT compilation, buffer
-- management, and host loops behind a simple configuration API.
--
-- Usage:
--
-- > ck <- compileSimpleKernel kernel logpdf
-- > (samples, diags) <- sampleChain ck [0.0, 0.0] $ burnIn 100 $ thin 2 $ defaultChainConfig { ccNumIterations = 1000 }
module HBayesian.Chain
  ( -- * Compiled kernels
    CompiledKernel
  , compileSimpleKernel
  , compileHMC
    -- * Chain configuration
  , ChainConfig (..)
  , defaultChainConfig
  , burnIn
  , thin
  , withSeed
  , verbose
    -- * Running chains
  , sampleChain
  , Diagnostic (..)
  ) where

import           Control.Monad       (when, zipWithM)
import           Data.Proxy          (Proxy (..))
import           Data.Word           (Word64)
import qualified Data.Vector.Storable as V

import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg (..), Module, TensorType)
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)
import           HHLO.Runtime.Buffer  (toDevice, toDeviceF32, fromDeviceF32)
import           HHLO.Runtime.Compile (compileWithOptions, defaultCompileOptions)
import           HHLO.Runtime.Execute (execute)
import           HHLO.Runtime.PJRT.Types (PJRTApi, PJRTClient, PJRTExecutable, PJRTBuffer,
                                           bufferTypeU64)

import           HBayesian.Core
import           HBayesian.HHLO.Ops hiding (map)
import           HBayesian.HHLO.PJRT
import           HBayesian.MCMC.HMC  (HMCState (..))

-----------------------------------------------------------------------------
-- CompiledKernel
-----------------------------------------------------------------------------

-- | Tag indicating which kind of step module was compiled.
data StepType = SimpleStep | HMCStep
  deriving (Eq, Show)

-- | A kernel that has been lowered to StableHLO modules but not yet
-- compiled to PJRT executables. The actual compilation happens inside
-- 'sampleChain' where the PJRT context is alive.
--
-- This design avoids the lifetime issue of PJRT handles: 'PJRTApi' and
-- 'PJRTClient' are raw pointers that become invalid when the plugin
-- is unloaded, so we cannot store compiled executables across calls.
data CompiledKernel = CompiledKernel
    { ckLdModule   :: !Module
    , ckGradModule :: !(Maybe Module)
    , ckStepModule :: !Module
    , ckShape      :: ![Int]
    , ckStepType   :: !StepType
    }

-- | Render a module and compile it via PJRT.
compileModule :: PJRTApi -> PJRTClient -> Module -> IO PJRTExecutable
compileModule api client modl =
    compileWithOptions api client (render modl) defaultCompileOptions

-- | Create a shape list from a 'KnownShape' proxy.
shapeList :: forall s. KnownShape s => [Int]
shapeList = Prelude.map fromIntegral (shapeVal (Proxy @s))

-- | Create a 'TensorType' from shape/dtype proxies.
tensorTypeOf :: forall s d. (KnownShape s, KnownDType d) => TensorType
tensorTypeOf = tensorType (Proxy @s) (Proxy @d)

-- | UI64 key type.
keyType :: TensorType
keyType = tensorType (Proxy @'[2]) (Proxy @'UI64)

-----------------------------------------------------------------------------
-- Buffer helpers
-----------------------------------------------------------------------------

bufferFromF32 :: PJRTApi -> PJRTClient -> [Int] -> [Float] -> IO PJRTBuffer
bufferFromF32 api client dims vals =
    toDeviceF32 api client (V.fromList vals) (Prelude.map fromIntegral dims)

bufferFromUI64 :: PJRTApi -> PJRTClient -> [Int] -> [Word64] -> IO PJRTBuffer
bufferFromUI64 api client dims vals =
    toDevice api client (V.fromList vals) (Prelude.map fromIntegral dims) bufferTypeU64

bufferToF32 :: PJRTApi -> PJRTBuffer -> Int -> IO [Float]
bufferToF32 api buf n = V.toList <$> fromDeviceF32 api buf n

-----------------------------------------------------------------------------
-- Compiling SimpleKernel (RandomWalk, EllipticalSlice)
-----------------------------------------------------------------------------

-- | Build a 'CompiledKernel' from a 'SimpleKernel'.
--
-- The log-posterior is compiled to a separate module so the host
-- can recompute log-density between steps.
compileSimpleKernel :: forall s d.
                       (KnownShape s, KnownDType d)
                    => SimpleKernel s d
                    -> (Tensor s d -> Builder (Tensor '[] d))
                    -> CompiledKernel
compileSimpleKernel kernel logpdf =
    let ldMod = moduleFromBuilder @'[] @d "main"
                  [FuncArg "theta" (tensorTypeOf @s @d)] $ do
          theta <- arg @s @d
          logpdf theta

        stepMod = moduleFromBuilder @s @d "main"
                    [ FuncArg "key" keyType
                    , FuncArg "pos" (tensorTypeOf @s @d)
                    , FuncArg "ld"  (tensorTypeOf @'[] @d)
                    ] $ do
          key <- arg @'[2] @'UI64
          pos <- arg @s @d
          ld  <- arg @'[] @d
          (state', _info) <- kernelStep kernel (Key key) (State pos ld)
          return (statePosition state')
    in CompiledKernel ldMod Nothing stepMod (shapeList @s) SimpleStep

-----------------------------------------------------------------------------
-- Compiling HMC kernels (HMC, MALA)
-----------------------------------------------------------------------------

-- | Build a 'CompiledKernel' from an HMC-style kernel.
--
-- Requires both the log-posterior and its gradient.
compileHMC :: forall s d info.
              (KnownShape s, KnownDType d)
           => Kernel s d (HMCState s d) info
           -> (Tensor s d -> Builder (Tensor '[] d))
           -> Gradient s d
           -> CompiledKernel
compileHMC kernel logpdf grad =
    let ldMod = moduleFromBuilder @'[] @d "main"
                  [FuncArg "theta" (tensorTypeOf @s @d)] $ do
          theta <- arg @s @d
          logpdf theta

        gradMod = moduleFromBuilder @s @d "main"
                    [FuncArg "theta" (tensorTypeOf @s @d)] $ do
          theta <- arg @s @d
          grad theta

        stepMod = moduleFromBuilder @s @d "main"
                    [ FuncArg "key" keyType
                    , FuncArg "pos" (tensorTypeOf @s @d)
                    , FuncArg "p"   (tensorTypeOf @s @d)
                    , FuncArg "ld"  (tensorTypeOf @'[] @d)
                    , FuncArg "g"   (tensorTypeOf @s @d)
                    ] $ do
          key <- arg @'[2] @'UI64
          pos <- arg @s @d
          p   <- arg @s @d
          ld  <- arg @'[] @d
          g   <- arg @s @d
          (state', _info) <- kernelStep kernel (Key key) (HMCState pos p ld g)
          return (hmcPosition state')
    in CompiledKernel ldMod (Just gradMod) stepMod (shapeList @s) HMCStep

-----------------------------------------------------------------------------
-- Chain configuration
-----------------------------------------------------------------------------

-- | Control parameters for a chain.
data ChainConfig = ChainConfig
    { ccNumIterations :: !Int
    , ccBurnIn        :: !Int
    , ccThinning      :: !Int
    , ccSeed          :: !Word64
    , ccVerbose       :: !Bool
    }

defaultChainConfig :: ChainConfig
defaultChainConfig = ChainConfig
    { ccNumIterations = 1000
    , ccBurnIn        = 0
    , ccThinning      = 1
    , ccSeed          = 42
    , ccVerbose       = False
    }

-- | Increase burn-in by N samples.
burnIn :: Int -> ChainConfig -> ChainConfig
burnIn n cfg = cfg { ccBurnIn = ccBurnIn cfg + n }

-- | Set thinning interval.
thin :: Int -> ChainConfig -> ChainConfig
thin n cfg = cfg { ccThinning = max 1 n }

-- | Override the PRNG seed.
withSeed :: Word64 -> ChainConfig -> ChainConfig
withSeed s cfg = cfg { ccSeed = s }

-- | Enable verbose progress output.
verbose :: ChainConfig -> ChainConfig
verbose cfg = cfg { ccVerbose = True }

-----------------------------------------------------------------------------
-- Diagnostics
-----------------------------------------------------------------------------

-- | A single-step diagnostic record.
data Diagnostic = Diagnostic
    { dStep       :: !Int
    , dAccepted   :: !Bool
    , dAcceptProb :: !Float
    }
    deriving (Show)

-----------------------------------------------------------------------------
-- Running a chain
-----------------------------------------------------------------------------

-- | Run a compiled kernel and return samples plus diagnostics.
--
-- The chain runs for @burnIn + numIterations * thinning@ steps total.
-- Samples are collected only after burn-in and only every @thinning@ steps.
--
-- This function opens a fresh PJRT context, compiles the modules,
-- executes the chain, and closes the context on return.
sampleChain :: CompiledKernel -> [Float] -> ChainConfig -> IO ([[Float]], [Diagnostic])
sampleChain ck pos0 cfg =
    withPJRTCPU $ \api client -> do
        let shape  = ckShape ck
            nDim   = product shape
            nTotal = ccNumIterations cfg
            nBurn  = ccBurnIn cfg
            thinBy = ccThinning cfg
            seed   = ccSeed cfg
            verb   = ccVerbose cfg
            totalSteps = nBurn + nTotal * thinBy

        -- Compile modules inside the PJRT context
        ldExe <- compileModule api client (ckLdModule ck)
        gradExe <- case ckGradModule ck of
            Nothing -> return Nothing
            Just gm -> Just <$> compileModule api client gm
        stepExe <- compileModule api client (ckStepModule ck)

        -- Evaluate initial log-density
        posBuf0 <- bufferFromF32 api client shape pos0
        [ldBuf0] <- execute api ldExe [posBuf0]
        ld0 <- head <$> bufferToF32 api ldBuf0 1

        -- Evaluate initial gradient (if HMC)
        g0 <- case gradExe of
            Nothing -> return (replicate nDim 0.0)
            Just gE -> do
                [gBuf0] <- execute api gE [posBuf0]
                bufferToF32 api gBuf0 nDim

        -- Run the chain
        (positions, diags) <- runLoop api client stepExe ldExe gradExe
                              (ckStepType ck) shape seed 0 pos0 ld0 g0 totalSteps verb

        -- Apply burn-in and thinning
        let postBurn = drop nBurn positions
            thinned  = take nTotal $ every thinBy postBurn
            diagsPost = drop nBurn diags
            diagsThin = take nTotal $ every thinBy diagsPost

        return (thinned, diagsThin)
  where
    every n xs = case xs of
        []     -> []
        (y:ys) -> y : every n (drop (n - 1) ys)

-- | Run N independent chains in parallel.
--
-- Each chain gets a distinct PRNG seed and optionally a perturbed
-- initial position. Results are returned in the same order as seeds.
parallelChains :: Int                          -- ^ number of chains
               -> ([Float] -> [Float])          -- ^ perturbation for initial values
               -> CompiledKernel                -- ^ compiled kernel
               -> [Float]                       -- ^ base initial position
               -> ChainConfig                   -- ^ chain configuration
               -> IO [([[Float]], [Diagnostic])]
parallelChains n perturb ck pos0 cfg =
    let seeds = [ccSeed cfg .. ccSeed cfg + fromIntegral n - 1]
        pos0s = pos0 : [perturb pos0 | _ <- [2..n]]
    in zipWithM (\s p -> sampleChain ck p (withSeed s cfg)) seeds pos0s

-- | Inner loop: run N steps, collecting all positions and diagnostics.
runLoop :: PJRTApi -> PJRTClient -> PJRTExecutable -> PJRTExecutable
        -> Maybe PJRTExecutable -> StepType -> [Int] -> Word64 -> Int
        -> [Float] -> Float -> [Float] -> Int -> Bool
        -> IO ([[Float]], [Diagnostic])
runLoop api client stepExe ldExe gradExe stepType shape seed step pos ld g n verb =
    go step pos ld g n []
  where
    nDim = product shape

    go _ _ _ _ 0 acc = return (reverse (map fst acc), reverse (map snd acc))
    go st p l gr remaining acc = do
        let key = [seed, fromIntegral st]

        keyBuf <- bufferFromUI64 api client [2] key
        posBuf <- bufferFromF32  api client shape p

        (newPos, newLD, newG, acceptProb) <- case stepType of
            SimpleStep -> do
                ldBuf <- bufferFromF32 api client [] [l]
                [newPosBuf] <- execute api stepExe [keyBuf, posBuf, ldBuf]
                newPos <- bufferToF32 api newPosBuf nDim
                -- Recompute log-density for next iteration
                [newLdBuf] <- execute api ldExe [newPosBuf]
                [newLd] <- bufferToF32 api newLdBuf 1
                let changed = newPos /= p
                let prob = if changed then 1.0 else 0.0
                return (newPos, newLd, replicate nDim 0.0, prob)

            HMCStep -> do
                pBuf <- bufferFromF32 api client shape (replicate nDim 0.0)
                ldBuf <- bufferFromF32 api client [] [l]
                gBuf <- bufferFromF32 api client shape gr
                [newPosBuf] <- execute api stepExe [keyBuf, posBuf, pBuf, ldBuf, gBuf]
                newPos <- bufferToF32 api newPosBuf nDim
                -- Recompute log-density and gradient for next iteration
                [newLdBuf] <- execute api ldExe [newPosBuf]
                [newLd] <- bufferToF32 api newLdBuf 1
                newG <- case gradExe of
                    Just gE -> do
                        [newGBuf] <- execute api gE [newPosBuf]
                        bufferToF32 api newGBuf nDim
                    Nothing -> return (replicate nDim 0.0)
                let changed = newPos /= p
                let prob = if changed then 1.0 else 0.0
                return (newPos, newLd, newG, prob)

        when verb $ do
            putStrLn $ "Step " ++ show st ++ ": pos=" ++ show newPos
                      ++ " accept=" ++ show acceptProb

        let diag = Diagnostic st (acceptProb > 0.5) acceptProb
        go (st + 1) newPos newLD newG (remaining - 1) ((newPos, diag) : acc)
