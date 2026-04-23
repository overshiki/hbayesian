{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.MCMC (tests) where

import           Data.Text           (Text)
import qualified Data.Text           as T
import           Test.Tasty
import           Test.Tasty.HUnit

import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)
import           HBayesian.Core
import           HBayesian.HHLO.Ops
import qualified HBayesian.HHLO.RNG as RNG
import           HBayesian.MCMC.EllipticalSlice
import           HBayesian.MCMC.HMC
import           HBayesian.MCMC.MALA

render1 :: forall s d. (KnownShape s, KnownDType d)
        => [FuncArg] -> Builder (Tensor s d) -> Text
render1 args b = render $ moduleFromBuilder @s @d "main" args b

-- | Dummy log-density for a 1-D standard normal.
--   log p(x) = -0.5 * x^2 (ignoring constant)
stdNormalLogPdf :: Tensor '[1] 'F32 -> Builder (Tensor '[] 'F32)
stdNormalLogPdf x = do
  xSq <- tmul x x
  negHalf <- constant @'[1] @'F32 (-0.5)
  negHalfXSq <- tmul negHalf xSq
  tsumAll negHalfXSq

-- | Dummy gradient for the 1-D standard normal.
--   d/dx (-0.5 * x^2) = -x
stdNormalGrad :: Gradient '[1] 'F32
stdNormalGrad x = do
  negOne <- constant @'[1] @'F32 (-1.0)
  tmul negOne x

tests :: TestTree
tests = testGroup "MCMC"
  [ testCase "RandomWalk kernelStep renders" $ do
      let k = Kernel
            { kernelInit = \_key pos -> do
                ld <- stdNormalLogPdf pos
                return (State pos ld)
            , kernelStep = \key state -> do
                (k1, k2) <- RNG.splitKey key
                let pos = statePosition state
                let ld = stateLogDensity state
                noise <- RNG.rngNormalF32 k1 >>= convert @'[1] @'F32 @'F32
                scaleT <- constant @'[1] @'F32 0.1
                scaledNoise <- tmul noise scaleT
                pos' <- tadd pos scaledNoise
                ld' <- stdNormalLogPdf pos'
                diff <- tsub ld' ld
                zero <- constant @'[] @'F32 0.0
                logAlpha <- tminimum diff zero
                u <- RNG.rngUniformF32 k2 >>= convert @'[] @'F32 @'F32
                logU <- tlog u
                accept <- tlessThan logU logAlpha
                acceptS <- tbroadcast @'[] @'[1] [] accept
                newPos <- tselect acceptS pos' pos
                newLd  <- tselect accept ld' ld
                one <- constant @'[] @'I64 1
                acceptProb <- texp logAlpha
                let info = Info acceptProb accept one
                return (State newPos newLd, info)
            }
      let mlir = render1 @'[1] @'F32
                   [ FuncArg "key" (TensorType [2] UI64)
                   , FuncArg "pos" (TensorType [1] F32)
                   , FuncArg "ld"  (TensorType [] F32)
                   ] $ do
            key <- arg @'[2] @'UI64
            pos <- arg @'[1] @'F32
            ld  <- arg @'[] @'F32
            (state', _info) <- kernelStep k (Key key) (State pos ld)
            return (statePosition state')
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains add" (T.isInfixOf "stablehlo.add" mlir)
      assertBool "contains subtract" (T.isInfixOf "stablehlo.subtract" mlir)
      assertBool "contains compare" (T.isInfixOf "stablehlo.compare" mlir)
      assertBool "contains select" (T.isInfixOf "stablehlo.select" mlir)

  , testCase "EllipticalSlice kernelStep renders" $ do
      let k = ellipticalSlice stdNormalLogPdf
      let mlir = render1 @'[1] @'F32
                   [ FuncArg "key" (TensorType [2] UI64)
                   , FuncArg "pos" (TensorType [1] F32)
                   , FuncArg "ld"  (TensorType [] F32)
                   ] $ do
            key <- arg @'[2] @'UI64
            pos <- arg @'[1] @'F32
            ld  <- arg @'[] @'F32
            (state', _info) <- kernelStep k (Key key) (State pos ld)
            return (statePosition state')
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains cosine" (T.isInfixOf "stablehlo.cosine" mlir)
      assertBool "contains sine" (T.isInfixOf "stablehlo.sine" mlir)
      assertBool "contains select" (T.isInfixOf "stablehlo.select" mlir)

  , testCase "HMC kernelStep renders" $ do
      let config = HMCConfig { hmcStepSize = 0.1, hmcNumLeapfrogSteps = 2 }
      let k = hmc stdNormalLogPdf stdNormalGrad config
      let mlir = render1 @'[1] @'F32
                   [ FuncArg "key" (TensorType [2] UI64)
                   , FuncArg "pos" (TensorType [1] F32)
                   , FuncArg "p"   (TensorType [1] F32)
                   , FuncArg "ld"  (TensorType [] F32)
                   , FuncArg "g"   (TensorType [1] F32)
                   ] $ do
            key <- arg @'[2] @'UI64
            pos <- arg @'[1] @'F32
            p   <- arg @'[1] @'F32
            ld  <- arg @'[] @'F32
            g   <- arg @'[1] @'F32
            (state', _info) <- kernelStep k (Key key) (HMCState pos p ld g)
            return (hmcPosition state')
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains add" (T.isInfixOf "stablehlo.add" mlir)
      assertBool "contains multiply" (T.isInfixOf "stablehlo.multiply" mlir)
      assertBool "contains select" (T.isInfixOf "stablehlo.select" mlir)

  , testCase "MALA kernelStep renders" $ do
      let config = MALAConfig { malaStepSize = 0.1 }
      let k = mala stdNormalLogPdf stdNormalGrad config
      let mlir = render1 @'[1] @'F32
                   [ FuncArg "key" (TensorType [2] UI64)
                   , FuncArg "pos" (TensorType [1] F32)
                   , FuncArg "p"   (TensorType [1] F32)
                   , FuncArg "ld"  (TensorType [] F32)
                   , FuncArg "g"   (TensorType [1] F32)
                   ] $ do
            key <- arg @'[2] @'UI64
            pos <- arg @'[1] @'F32
            p   <- arg @'[1] @'F32
            ld  <- arg @'[] @'F32
            g   <- arg @'[1] @'F32
            (state', _info) <- kernelStep k (Key key) (HMCState pos p ld g)
            return (hmcPosition state')
      assertBool "contains rng_bit_generator" (T.isInfixOf "stablehlo.rng_bit_generator" mlir)
      assertBool "contains add" (T.isInfixOf "stablehlo.add" mlir)
      assertBool "contains multiply" (T.isInfixOf "stablehlo.multiply" mlir)
      assertBool "contains select" (T.isInfixOf "stablehlo.select" mlir)
  ]
