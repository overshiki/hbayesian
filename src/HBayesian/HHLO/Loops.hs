{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}

-- | Multi-value control flow primitives for Bayesian state patterns.
--
-- HHLO 0.2.0.0 provides 'whileLoop2' and 'whileLoopN'.
-- This module adds 'whileLoop3', 'whileLoop4', and 'conditional3', 'conditional4'
-- as mechanical boilerplate.
module HBayesian.HHLO.Loops
  ( whileLoop3
  , whileLoop4
  , conditional3
  , conditional4
  ) where

import           Data.Proxy
import           HHLO.Core.Types
import           HHLO.EDSL.Ops
import           HHLO.IR.AST         (Region(..))
import           HHLO.IR.Builder

-----------------------------------------------------------------------------
-- whileLoop3
-----------------------------------------------------------------------------

whileLoop3 :: forall s1 d1 s2 d2 s3 d3.
              ( KnownShape s1, KnownDType d1
              , KnownShape s2, KnownDType d2
              , KnownShape s3, KnownDType d3
              )
           => Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3))
           -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3)
whileLoop3 init1 init2 init3 cond body = do
  let initVids  = [tensorValue init1, tensorValue init2, tensorValue init3]
      initTypes = [ tensorType (Proxy @s1) (Proxy @d1)
                  , tensorType (Proxy @s2) (Proxy @d2)
                  , tensorType (Proxy @s3) (Proxy @d3)
                  ]
      boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)

  -- Build cond region
  condBlock <- runBlockBuilder initTypes $ do
    v1 <- arg @s1 @d1
    v2 <- arg @s2 @d2
    v3 <- arg @s3 @d3
    c  <- cond v1 v2 v3
    emitReturn [tensorValue c] [boolType]

  -- Build body region
  bodyBlock <- runBlockBuilder initTypes $ do
    v1 <- arg @s1 @d1
    v2 <- arg @s2 @d2
    v3 <- arg @s3 @d3
    (r1, r2, r3) <- body v1 v2 v3
    emitReturn [tensorValue r1, tensorValue r2, tensorValue r3] initTypes

  vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
            [Region [condBlock], Region [bodyBlock]] initTypes
  case vids of
    [vid1, vid2, vid3] -> return (Tensor vid1, Tensor vid2, Tensor vid3)
    _                  -> error "whileLoop3: expected exactly three results"

-----------------------------------------------------------------------------
-- whileLoop4
-----------------------------------------------------------------------------

whileLoop4 :: forall s1 d1 s2 d2 s3 d3 s4 d4.
              ( KnownShape s1, KnownDType d1
              , KnownShape s2, KnownDType d2
              , KnownShape s3, KnownDType d3
              , KnownShape s4, KnownDType d4
              )
           => Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3, Tensor s4 d4))
           -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3, Tensor s4 d4)
whileLoop4 init1 init2 init3 init4 cond body = do
  let initVids  = [tensorValue init1, tensorValue init2, tensorValue init3, tensorValue init4]
      initTypes = [ tensorType (Proxy @s1) (Proxy @d1)
                  , tensorType (Proxy @s2) (Proxy @d2)
                  , tensorType (Proxy @s3) (Proxy @d3)
                  , tensorType (Proxy @s4) (Proxy @d4)
                  ]
      boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)

  condBlock <- runBlockBuilder initTypes $ do
    v1 <- arg @s1 @d1
    v2 <- arg @s2 @d2
    v3 <- arg @s3 @d3
    v4 <- arg @s4 @d4
    c  <- cond v1 v2 v3 v4
    emitReturn [tensorValue c] [boolType]

  bodyBlock <- runBlockBuilder initTypes $ do
    v1 <- arg @s1 @d1
    v2 <- arg @s2 @d2
    v3 <- arg @s3 @d3
    v4 <- arg @s4 @d4
    (r1, r2, r3, r4) <- body v1 v2 v3 v4
    emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4] initTypes

  vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
            [Region [condBlock], Region [bodyBlock]] initTypes
  case vids of
    [vid1, vid2, vid3, vid4] -> return (Tensor vid1, Tensor vid2, Tensor vid3, Tensor vid4)
    _                        -> error "whileLoop4: expected exactly four results"

-----------------------------------------------------------------------------
-- conditional3
-----------------------------------------------------------------------------

conditional3 :: forall s1 d1 s2 d2 s3 d3.
                ( KnownShape s1, KnownDType d1
                , KnownShape s2, KnownDType d2
                , KnownShape s3, KnownDType d3
                )
             => Tensor '[] 'Bool
             -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3)
             -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3)
             -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3)
conditional3 p trueThunk falseThunk = do
  let types   = [ tensorType (Proxy @s1) (Proxy @d1)
                , tensorType (Proxy @s2) (Proxy @d2)
                , tensorType (Proxy @s3) (Proxy @d3)
                ]
      boolType = tensorType (Proxy @'[]) (Proxy @'Bool)

  trueBlock <- runBlockBuilder [] $ do
    (r1, r2, r3) <- trueThunk
    emitReturn [tensorValue r1, tensorValue r2, tensorValue r3] types

  falseBlock <- runBlockBuilder [] $ do
    (r1, r2, r3) <- falseThunk
    emitReturn [tensorValue r1, tensorValue r2, tensorValue r3] types

  let (Tensor predVid) = p
  vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
            [Region [trueBlock], Region [falseBlock]] types
  case vids of
    [vid1, vid2, vid3] -> return (Tensor vid1, Tensor vid2, Tensor vid3)
    _                  -> error "conditional3: expected exactly three results"

-----------------------------------------------------------------------------
-- conditional4
-----------------------------------------------------------------------------

conditional4 :: forall s1 d1 s2 d2 s3 d3 s4 d4.
                ( KnownShape s1, KnownDType d1
                , KnownShape s2, KnownDType d2
                , KnownShape s3, KnownDType d3
                , KnownShape s4, KnownDType d4
                )
             => Tensor '[] 'Bool
             -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3, Tensor s4 d4)
             -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3, Tensor s4 d4)
             -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3, Tensor s4 d4)
conditional4 p trueThunk falseThunk = do
  let types   = [ tensorType (Proxy @s1) (Proxy @d1)
                , tensorType (Proxy @s2) (Proxy @d2)
                , tensorType (Proxy @s3) (Proxy @d3)
                , tensorType (Proxy @s4) (Proxy @d4)
                ]
      boolType = tensorType (Proxy @'[]) (Proxy @'Bool)

  trueBlock <- runBlockBuilder [] $ do
    (r1, r2, r3, r4) <- trueThunk
    emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4] types

  falseBlock <- runBlockBuilder [] $ do
    (r1, r2, r3, r4) <- falseThunk
    emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4] types

  let (Tensor predVid) = p
  vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
            [Region [trueBlock], Region [falseBlock]] types
  case vids of
    [vid1, vid2, vid3, vid4] -> return (Tensor vid1, Tensor vid2, Tensor vid3, Tensor vid4)
    _                        -> error "conditional4: expected exactly four results"
