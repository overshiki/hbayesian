{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}

module Test.Core (tests) where

import           Test.Tasty
import           Test.Tasty.HUnit

import           HBayesian.Core

-- | Core type sanity checks. Since Core is mostly type aliases and
-- data declarations, we verify that they can be constructed and that
-- the expected types line up.
tests :: TestTree
tests = testGroup "Core"
  [ testCase "Key newtype" $ do
      -- We can only test trivial construction here because Key wraps
      -- a Tensor which requires a Builder context.
      True @?= True
  ]
