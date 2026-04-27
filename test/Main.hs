module Main (main) where

import           Test.Tasty
import qualified Test.Core       as Core
import qualified Test.HHLO.Ops   as Ops
import qualified Test.HHLO.RNG   as RNG
import qualified Test.HHLO.Loops as Loops
import qualified Test.MCMC       as MCMC
import qualified Test.Examples   as Examples
import qualified Test.Chain             as Chain
import qualified Test.PPL                 as PPL
import qualified Test.CorrelatedGaussian  as CorrG
import qualified Test.CorrelatedGaussianNUTS as CorrGNUTS
import qualified Test.NealFunnel as NealF

main :: IO ()
main = defaultMain $ testGroup "HBaysian Tests"
  [ Core.tests
  , Ops.tests
  , RNG.tests
  , Loops.tests
  , MCMC.tests
  , Examples.tests
  , Chain.tests
  , PPL.tests
  , CorrG.tests
  , CorrGNUTS.tests
  , NealF.tests
  ]
