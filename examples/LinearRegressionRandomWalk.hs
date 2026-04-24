{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 1: Bayesian Linear Regression with RandomWalk MH.
--
-- Model: y_i = alpha + beta * x_i + epsilon_i,  epsilon_i ~ N(0, 0.25)
-- Prior:  alpha ~ N(0, 1),  beta ~ N(0, 1)
module LinearRegressionRandomWalk
  ( dataset
  , linearRegLogPdf
  , makeKernel
  , renderStepMlir
  , runChain
  , runChainV2
  ) where

import           Data.Word           (Word64)
import           Data.Text           (Text)
import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import           HBayesian.HHLO.PJRT
import           HBayesian.MCMC.RandomWalk
import           HBayesian.Chain
import           Common

-- | Fixed synthetic dataset (n = 5).
dataset :: [(Float, Float)]
dataset =
  [ (0.0,  0.5)
  , (1.0,  2.0)
  , (2.0,  3.5)
  , (3.0,  5.0)
  , (4.0,  6.5)
  ]

-- | Log-posterior for Bayesian linear regression.
linearRegLogPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
linearRegLogPdf theta = do
  alpha <- tslice1 @2 @'F32 theta 0
  beta  <- tslice1 @2 @'F32 theta 1

  let likelihoodPoint (x, y) = do
        xT <- tconstant @'[] @'F32 (realToFrac x)
        yT <- tconstant @'[] @'F32 (realToFrac y)
        betaX <- tmul beta xT
        predVal <- tadd alpha betaX
        diff <- tsub yT predVal
        diffSq <- tmul diff diff
        negTwo <- tconstant @'[] @'F32 (-2.0)
        tmul negTwo diffSq

  llh0 <- likelihoodPoint (dataset !! 0)
  llh1 <- likelihoodPoint (dataset !! 1)
  llh2 <- likelihoodPoint (dataset !! 2)
  llh3 <- likelihoodPoint (dataset !! 3)
  llh4 <- likelihoodPoint (dataset !! 4)

  llh01 <- tadd llh0 llh1
  llh23 <- tadd llh2 llh3
  llh0123 <- tadd llh01 llh23
  llh <- tadd llh0123 llh4

  alphaSq <- tmul alpha alpha
  betaSq  <- tmul beta beta
  negHalf <- tconstant @'[] @'F32 (-0.5)
  priorAlpha <- tmul negHalf alphaSq
  priorBeta  <- tmul negHalf betaSq

  tadd llh =<< tadd priorAlpha priorBeta

-- | Factory: build a RandomWalk kernel for this model.
makeKernel :: RWConfig -> SimpleKernel '[2] 'F32
makeKernel config = randomWalk linearRegLogPdf config

-- | Tier A: render one kernel step to MLIR text.
renderStepMlir :: Text
renderStepMlir =
  renderKernelStep @'[2] @'F32
    [ FuncArg "key" (TensorType [2] UI64)
    , FuncArg "pos" (TensorType [2] F32)
    , FuncArg "ld"  (TensorType [] F32)
    ] $ do
      key <- arg @'[2] @'UI64
      pos <- arg @'[2] @'F32
      ld  <- arg @'[] @'F32
      (state', _info) <- kernelStep (makeKernel (RWConfig 0.1)) (Key key) (State pos ld)
      return (statePosition state')

-- | Tier B: run a short chain on PJRT and return the sampled positions.
runChain :: IO [[Float]]
runChain = withPJRTCPU $ \api client -> do
    let kernel = makeKernel (RWConfig 0.1)

    -- Compile the log-pdf module
    let ldMod = moduleFromBuilder @'[] @'F32 "main"
                  [ FuncArg "pos" (TensorType [2] F32) ] $ do
          pos <- arg @'[2] @'F32
          linearRegLogPdf pos
    ldExe <- compileModule api client ldMod

    -- Compile the kernel-step module (single result: position)
    let stepMod = moduleFromBuilder @'[2] @'F32 "main"
                    [ FuncArg "key" (TensorType [2] UI64)
                    , FuncArg "pos" (TensorType [2] F32)
                    , FuncArg "ld"  (TensorType [] F32)
                    ] $ do
          key <- arg @'[2] @'UI64
          pos <- arg @'[2] @'F32
          ld  <- arg @'[] @'F32
          (state', _info) <- kernelStep kernel (Key key) (State pos ld)
          return (statePosition state')
    stepExe <- compileModule api client stepMod

    let seed :: Word64 = 42
        pos0 = [0.0, 0.0]

    -- Compute initial log-density
    posBuf0 <- bufferFromF32 api client [2] pos0
    [ldBuf0] <- executeModule api ldExe [posBuf0]
    [ld0] <- bufferToF32 api ldBuf0 1

    loop api client stepExe ldExe seed (0 :: Int) pos0 ld0 (10 :: Int) []
  where
    loop _ _ _ _ _ _ _ _ 0 acc = return (reverse acc)
    loop api client stepExe ldExe seed step pos ld n acc = do
        let key = [seed, fromIntegral step]
        keyBuf <- bufferFromUI64 api client [2] key
        posBuf <- bufferFromF32 api client [2] pos
        ldBuf  <- bufferFromF32 api client [] [ld]
        [newPosBuf] <- executeModule api stepExe [keyBuf, posBuf, ldBuf]
        newPos <- bufferToF32 api newPosBuf 2
        -- Recompute log-density for the next step
        [newLdBuf] <- executeModule api ldExe [newPosBuf]
        [newLd] <- bufferToF32 api newLdBuf 1
        loop api client stepExe ldExe seed (step + 1) newPos newLd (n - 1) (newPos : acc)

-- | v0.2: Run a chain using the 'Chain' combinators.
runChainV2 :: IO ([[Float]], [Diagnostic])
runChainV2 = do
    let kernel = makeKernel (RWConfig 0.1)
        ck     = compileSimpleKernel kernel linearRegLogPdf
    sampleChain ck [0.0, 0.0] $ defaultChainConfig
        { ccNumIterations = 10
        , ccSeed = 42
        }
