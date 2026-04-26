{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 2: Gaussian Process Regression with Elliptical Slice Sampling.
--
-- We use a tiny n=3 dataset with an identity covariance prior
-- (K = I, so K^{-1} = I) to avoid needing a Cholesky decomposition
-- in the Builder.  The example still demonstrates ESS on a Gaussian
-- prior, which is the sampler's natural habitat.
module GaussianProcessEllipticalSlice
  ( yData
  , gpLogPdf
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
import           HBayesian.MCMC.EllipticalSlice
import           HBayesian.Chain

-- | Observed responses (n = 3).
yData :: [Float]
yData = [0.5, 2.0, 3.5]

-- | GP log-posterior with identity prior covariance.
gpLogPdf :: Tensor '[3] 'F32 -> Builder (Tensor '[] 'F32)
gpLogPdf f = do
  y0 <- tconstant @'[] @'F32 (realToFrac (yData !! 0))
  y1 <- tconstant @'[] @'F32 (realToFrac (yData !! 1))
  y2 <- tconstant @'[] @'F32 (realToFrac (yData !! 2))

  f0 <- tslice1 @3 @'F32 f 0
  f1 <- tslice1 @3 @'F32 f 1
  f2 <- tslice1 @3 @'F32 f 2

  d0 <- tsub y0 f0
  d1 <- tsub y1 f1
  d2 <- tsub y2 f2

  d0sq <- tmul d0 d0
  d1sq <- tmul d1 d1
  d2sq <- tmul d2 d2

  negHalf <- tconstant @'[] @'F32 (-0.5)
  llh0 <- tmul negHalf d0sq
  llh1 <- tmul negHalf d1sq
  llh2 <- tmul negHalf d2sq

  llh01 <- tadd llh0 llh1
  llh   <- tadd llh01 llh2

  f0sq <- tmul f0 f0
  f1sq <- tmul f1 f1
  f2sq <- tmul f2 f2

  fSqSum <- tadd f0sq =<< tadd f1sq f2sq
  prior <- tmul negHalf fSqSum

  tadd llh prior

-- | Factory: build an Elliptical Slice kernel for this model.
makeKernel :: SimpleKernel '[3] 'F32
makeKernel = ellipticalSlice gpLogPdf

-- | Tier A: render one kernel step to MLIR text.
renderStepMlir :: Text
renderStepMlir =
  renderKernelStep @'[3] @'F32
    [ FuncArg "key" (TensorType [2] UI64)
    , FuncArg "pos" (TensorType [3] F32)
    , FuncArg "ld"  (TensorType [] F32)
    ] $ do
      key <- arg @'[2] @'UI64
      pos <- arg @'[3] @'F32
      ld  <- arg @'[] @'F32
      (state', _info) <- kernelStep makeKernel (Key key) (State pos ld)
      return (statePosition state')

-- | Tier B: run a short chain on PJRT and return the sampled latent vectors.
runChain :: IO [[Float]]
runChain = withPJRTCPU $ \api client -> do
    -- Compile the log-pdf module
    let ldMod = moduleFromBuilder @'[] @'F32 "main"
                  [ FuncArg "f" (TensorType [3] F32) ] $ do
          f <- arg @'[3] @'F32
          gpLogPdf f
    ldExe <- compileModule api client ldMod

    -- Compile the kernel-step module (single result: position)
    let stepMod = moduleFromBuilder @'[3] @'F32 "main"
                    [ FuncArg "key" (TensorType [2] UI64)
                    , FuncArg "pos" (TensorType [3] F32)
                    , FuncArg "ld"  (TensorType [] F32)
                    ] $ do
          key <- arg @'[2] @'UI64
          pos <- arg @'[3] @'F32
          ld  <- arg @'[] @'F32
          (state', _info) <- kernelStep makeKernel (Key key) (State pos ld)
          return (statePosition state')
    stepExe <- compileModule api client stepMod

    let seed :: Word64 = 42
        f0   = [0.0, 0.0, 0.0]

    -- Compute initial log-density
    fBuf0 <- bufferFromF32 api client [3] f0
    [ldBuf0] <- executeModule api ldExe [fBuf0]
    [ld0] <- bufferToF32 api ldBuf0 1

    loop api client stepExe ldExe seed (0 :: Int) f0 ld0 (10 :: Int) []
  where
    loop _ _ _ _ _ _ _ _ 0 acc = return (reverse acc)
    loop api client stepExe ldExe seed step pos ld n acc = do
        let key = [seed, fromIntegral step]
        keyBuf <- bufferFromUI64 api client [2] key
        posBuf <- bufferFromF32 api client [3] pos
        ldBuf  <- bufferFromF32 api client [] [ld]
        [newPosBuf] <- executeModule api stepExe [keyBuf, posBuf, ldBuf]
        newPos <- bufferToF32 api newPosBuf 3
        [newLdBuf] <- executeModule api ldExe [newPosBuf]
        [newLd] <- bufferToF32 api newLdBuf 1
        loop api client stepExe ldExe seed (step + 1) newPos newLd (n - 1) (newPos : acc)

-- | v0.2: Run a chain using the 'Chain' combinators.
runChainV2 :: IO ([[Float]], [Diagnostic])
runChainV2 = do
    let ck = compileSimpleKernel makeKernel gpLogPdf
    sampleChain ck [0.0, 0.0, 0.0] $ defaultChainConfig
        { ccNumIterations = 10
        , ccSeed = 42
        }
