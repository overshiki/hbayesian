{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE OverloadedStrings  #-}
{-# LANGUAGE TypeApplications #-}

module Test.PPL (tests) where

import           Data.List           (isInfixOf)
import           Test.Tasty
import           Test.Tasty.HUnit

import           HHLO.Core.Types
import           HHLO.IR.AST         (FuncArg(..), TensorType(..))
import           HHLO.IR.Builder
import           HHLO.IR.Pretty      (render)

import           HBayesian.Core
import           HBayesian.HHLO.Ops hiding (map)
import           HBayesian.PPL
import           HBayesian.MCMC.RandomWalk
import           HBayesian.Chain

-- | A tiny PPL model: two parameters with normal priors and one observation.
tinyModel :: PPL 2 ()
tinyModel = do
    alpha <- param 0
    beta  <- param 1
    observe "alpha_prior" (normal 0.0 1.0) alpha
    observe "beta_prior"  (normal 0.0 1.0) beta
    -- observation: y = 1.0 at x = 0.5, likelihood N(alpha + beta * 0.5, 0.5)
    x <- liftBuilder $ tconstant 0.5
    mu <- liftBuilder $ tadd alpha =<< tmul beta x
    y <- liftBuilder $ tconstant 1.0
    sigma <- liftBuilder $ tconstant 0.5
    observe "y" (normalT mu sigma) y

-- | The same model written manually as a log-posterior.
tinyLogPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
tinyLogPdf = runPPL tinyModel

-- | Render the PPL-derived log-posterior to MLIR for inspection.
tinyMlir :: String
tinyMlir = show $ render $ moduleFromBuilder @'[] @'F32 "main"
    [FuncArg "theta" (TensorType [2] F32)] $ do
        theta <- arg @'[2] @'F32
        tinyLogPdf theta

tests :: TestTree
tests = testGroup "PPL"
    [ testCase "PPL-derived model renders non-empty MLIR" $ do
        assertBool "MLIR should be non-empty" (not (null tinyMlir))

    , testCase "PPL-derived model contains expected ops" $ do
        assertBool "should contain stablehlo.constant" ("stablehlo.constant" `isInfixOf` tinyMlir)
        assertBool "should contain stablehlo.add" ("stablehlo.add" `isInfixOf` tinyMlir)
        assertBool "should contain stablehlo.multiply" ("stablehlo.multiply" `isInfixOf` tinyMlir)
    ]
