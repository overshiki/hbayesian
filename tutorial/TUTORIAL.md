# HBayesian Tutorial

> **Updated for v0.2.** This tutorial covers both the original v0.1 APIs (kernels, manual buffer management) and the new v0.2 abstractions (Chain combinators, PPL layer, diagnostics).

This tutorial has two levels.

- **Level 1** is for *users* who want to run Bayesian inference on their models. You will learn the ideas, the procedure, and the meaning of every public function you call.
- **Level 2** is for *algorithm designers* who want to implement new samplers. You will learn the low-level `Builder` API, the PRNG system, and the full path from a Haskell function to a PJRT executable.

At the end we reflect on the API itself: what feels natural, what feels mechanical, and how the interface could evolve.

---

## Level 1 — Using Inference Algorithms

### 1.1 The big picture

Bayesian inference means: *given some data and a model, draw samples from the posterior distribution of the model parameters.*

In HBayesian this is always done with **MCMC** (Markov Chain Monte Carlo). The procedure is:

1. Write a **log-posterior** function `p(θ | data)`.
2. Choose a **sampler** (RandomWalk, EllipticalSlice, HMC, MALA).
3. Run a **chain**: starting from an initial guess, the sampler repeatedly proposes new parameter values and decides whether to keep them.
4. Collect the **samples** and use them for estimation, prediction, or visualization.

Every sampler in HBayesian is a pure function that compiles to a self-contained StableHLO module and executes on PJRT (CPU, GPU, or TPU). There is no "interpreter mode". The chain itself runs on the device; the host only drives the loop.

---

### 1.2 Your first chain: Bayesian linear regression with RandomWalk

Here is the smallest complete program that actually samples.

```haskell
{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified LinearRegressionRandomWalk as Ex

main :: IO ()
main = do
    samples <- Ex.runChain       -- IO [[Float]]
    putStrLn "First sample:"
    print (head samples)
    putStrLn "Last sample:"
    print (last samples)
```

Run it:

```bash
cabal run hbayesian-examples -- --execute
```

You will see 10 parameter vectors printed. Each vector `[alpha, beta]` is a draw from the posterior of the Bayesian linear regression model. The chain starts near `[0,0]` and gradually drifts toward the region favoured by the data.

---

### 1.3 The semantics of the interface

#### `logPdf :: Tensor s d -> Builder (Tensor '[] d)`

This is your model. It takes a parameter tensor `θ` and returns a scalar: the unnormalised log-posterior `log p(θ | data)`. The `Builder` monad means you are not computing the value directly; you are building a *graph* of tensor operations that will later be compiled to XLA.

```haskell
linearRegLogPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
linearRegLogPdf theta = do
    alpha <- tslice1 @2 @'F32 theta 0
    beta  <- tslice1 @2 @'F32 theta 1
    -- likelihood + prior ...
```

Key point: `linearRegLogPdf` is just a Haskell function. You can use ordinary Haskell to unroll loops, look up constants from lists, or compose sub-expressions. But every operation *inside* the `do` block must be an HHLO primitive (`tslice1`, `tadd`, `tmul`, etc.).

#### `Kernel s d state info`

A sampler is a record with two fields:

```haskell
data Kernel s d state info = Kernel
  { kernelInit :: Key -> Tensor s d -> Builder (state s d)
  , kernelStep :: Key -> state s d -> Builder (state s d, info s d)
  }
```

- `kernelInit` — given a PRNG key and an initial position, creates the sampler's internal state (which may include momentum, gradient cache, etc.).
- `kernelStep` — given a PRNG key and the current state, produces a new state and some diagnostic information.

The type parameters are:
- `s` — shape of the parameter vector (e.g. `'[2]`)
- `d` — element dtype (almost always `'F32`)
- `state` — sampler-specific state (e.g. `State` for RandomWalk, `HMCState` for HMC)
- `info` — diagnostic output (acceptance probability, accept/reject flag, etc.)

#### `randomWalk :: (Tensor s d -> Builder (Tensor '[] d)) -> RWConfig -> SimpleKernel s d`

`randomWalk` takes your log-posterior and a configuration, and returns a `Kernel` specialised to the RandomWalk Metropolis-Hastings algorithm. `SimpleKernel` is just a type synonym for `Kernel s d State Info` — the simplest possible state (position + log-density) and info (acceptance probability + flag + step count).

```haskell
makeKernel :: RWConfig -> SimpleKernel '[2] 'F32
makeKernel config = randomWalk linearRegLogPdf config
```

#### `runChain :: IO [[Float]]`

This is *not* part of the library core. It lives in the example modules and shows you the complete execution pattern. Conceptually it does:

1. Load the PJRT plugin.
2. Compile the log-posterior to a PJRT executable.
3. Compile `kernelStep` to a PJRT executable (wrapped to return a single result).
4. Create buffers for the key, position, log-density, etc.
5. Loop N times: execute step → read back position → recompute log-density → feed into next iteration.
6. Return the list of sampled positions.

You can copy `runChain` into your own project and adapt the shapes, initial values, and loop length.

---

### 1.4 Configuring a sampler

Each sampler has its own configuration type:

| Sampler | Config | Tunable parameters |
|---------|--------|-------------------|
| RandomWalk | `RWConfig { rwScale :: Double }` | Proposal standard deviation |
| EllipticalSlice | *none* | — |
| HMC | `HMCConfig { hmcStepSize :: Double, hmcNumLeapfrogSteps :: Int }` | Leapfrog step size and trajectory length |
| MALA | `MALAConfig { malaStepSize :: Double }` | Single leapfrog step size |

**RandomWalk** is the easiest to understand but hardest to tune. If `rwScale` is too small, the chain explores slowly; too large, and most proposals are rejected. A good rule of thumb is to aim for an acceptance rate around 20–40%.

**EllipticalSlice** is attractive because it has *no tunable parameters*. It is designed for models with Gaussian priors and tends to work well when the likelihood is not too pathological.

**HMC** is the workhorse for high-dimensional problems. It uses gradient information to make informed proposals. The trade-off is between `hmcStepSize` (larger = faster exploration but more rejections) and `hmcNumLeapfrogSteps` (longer trajectories = lower correlation between samples but higher cost per step).

**MALA** is essentially HMC with a single leapfrog step. It is cheaper per iteration but the samples are more correlated.

---

### 1.5 Working with your own model

To run inference on your own problem, you need to provide three things:

1. **The log-posterior** `Tensor s 'F32 -> Builder (Tensor '[] 'F32)`.
2. **The initial position** — a list of `Float`s with the same length as the shape `s`.
3. **Optionally, the gradient** — if you use HMC or MALA.

Here is a template:

```haskell
{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}

module MyInference where

import HBayesian.Core
import HBayesian.HHLO.Ops
import HBayesian.HHLO.PJRT
import HBayesian.MCMC.RandomWalk
import HHLO.IR.Builder
import Common

-- 1. Define the log-posterior
myLogPdf :: Tensor '[3] 'F32 -> Builder (Tensor '[] 'F32)
myLogPdf theta = do
    -- your model here
    return logP

-- 2. Build the kernel
myKernel :: SimpleKernel '[3] 'F32
myKernel = randomWalk myLogPdf (RWConfig 0.05)

-- 3. Run a chain (adapt from LinearRegressionRandomWalk.runChain)
runMyChain :: IO [[Float]]
runMyChain = withPJRTCPU $ \api client -> do
    -- compile modules, create buffers, loop ...
    return samples
```

The example modules in `examples/` are the best reference. Copy the one whose sampler matches your needs and replace the log-posterior, shapes, and initial values.

---

### 1.6 Sampler selection guide

| Situation | Recommended sampler | Why |
|-----------|---------------------|-----|
| Low dimension (≤5), no gradient | RandomWalk | Simple, easy to implement |
| Gaussian prior, any dimension | EllipticalSlice | No tuning, naturally suited to Gaussian geometry |
| High dimension, gradient available | HMC | Informed proposals, good mixing |
| High dimension, gradient available, cheap iterations | MALA | Simpler than HMC, still uses gradient |
| Pathological geometry (strong correlations) | HMC with small step size | RandomWalk and ESS may get stuck |

---

## Level 2 — Designing Inference Algorithms

Level 1 treated the sampler as a black box. Level 2 opens the box. You will learn how to write a new MCMC algorithm from scratch using the `Builder` API.

### 2.1 The type system: `Tensor`, `Builder`, and `Key`

#### `Tensor s d`

A `Tensor` is not a container of numbers. It is a *node* in a computational graph. The type parameters guarantee shape and dtype safety at compile time:

```haskell
x :: Tensor '[3] 'F32       -- a 1-D float vector of length 3
y :: Tensor '[] 'F32        -- a scalar float
z :: Tensor '[2, 3] 'F32   -- a 2x3 matrix
```

You cannot accidentally add a `[3]` tensor to a `[2]` tensor — the types will not match.

#### `Builder a`

`Builder` is the monad in which you construct StableHLO graphs. Every HHLO primitive returns a `Builder` action:

```haskell
tadd :: Tensor s d -> Tensor s d -> Builder (Tensor s d)
tconstant :: forall s d. (KnownShape s, KnownDType d) => Float -> Builder (Tensor s d)
```

When you write:

```haskell
myExpr = do
    a <- tconstant @'[2] @'F32 1.0
    b <- tconstant @'[2] @'F32 2.0
    tadd a b
```

you are not computing `3.0`. You are building a graph that, when compiled and executed, will produce `3.0`. This is the essence of the HHLO EDSL: Haskell is the meta-language, StableHLO is the target language.

#### `Key`

PRNG state is explicit and immutable:

```haskell
newtype Key = Key (Tensor '[2] 'UI64)
```

A `Key` is a 2-element `Word64` tensor. To get random numbers, you must first *split* the key:

```haskell
splitKey :: Key -> Builder (Key, Key)
```

This produces two statistically independent keys. One is consumed by a random operation; the other is passed forward to the next step. This is the **SplittableRandom** discipline — there is no global RNG state.

```haskell
myRandomOp key = do
    (key1, key2) <- splitKey key
    noise <- rngNormalF32 key1         -- consumes key1
    -- ... use noise ...
    return (result, key2)              -- key2 lives on
```

---

### 2.2 Anatomy of a `Kernel`

Every sampler follows the same pattern:

```haskell
data Kernel s d state info = Kernel
  { kernelInit :: Key -> Tensor s d -> Builder (state s d)
  , kernelStep :: Key -> state s d -> Builder (state s d, info s d)
  }
```

#### `kernelInit`

Takes a PRNG key (usually ignored for deterministic initialisation) and an initial position, and builds the sampler's full state. For RandomWalk this is trivial:

```haskell
kernelInit _key pos = do
    ld <- logpdf pos
    return $ State pos ld
```

For HMC it is more involved: you must also sample initial momentum from a standard normal and evaluate the gradient:

```haskell
kernelInit key pos = do
    ld <- logpdf pos
    g  <- grad pos
    zeroM <- tconstant 0.0
    return $ HMCState pos zeroM ld g
```

#### `kernelStep`

This is where the algorithm lives. It receives a *fresh* key and the current state, and must produce a new state. The typical structure is:

1. **Split the key** into sub-keys for different random operations.
2. **Propose** a new state using randomness.
3. **Evaluate** the acceptance criterion (Metropolis ratio, slice variable, Hamiltonian difference, etc.).
4. **Decide** accept or reject using a uniform random draw.
5. **Select** the new state or keep the old one.
6. **Return** the chosen state and diagnostic info.

---

### 2.3 Implementing a custom sampler from scratch

Let us implement a **deterministic-scan Metropolis-within-Gibbs** sampler for a 2-D problem. This is not in the library; we write it to demonstrate every API surface.

The idea: update one coordinate at a time. For coordinate `i`, propose `θ_i' = θ_i + ε * N(0,1)` and accept/reject with the standard Metropolis ratio. This is simpler than full RandomWalk because each proposal is 1-D, but the structure is the same.

```haskell
{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Tutorial.CustomSampler where

import HHLO.Core.Types
import HHLO.IR.Builder
import HBayesian.Core
import HBayesian.HHLO.Ops
import qualified HBayesian.HHLO.RNG as RNG

-- Configuration
newtype ScanConfig = ScanConfig { scanScale :: Double }

-- State: just position and log-density (same as RandomWalk)
data ScanState s d = ScanState
  { scanPosition  :: !(Tensor s d)
  , scanLogDens   :: !(Tensor '[] d)
  }

-- Info: same as standard Info
type ScanInfo s d = Info s d

-- The sampler
scanSampler :: forall s d.
               (KnownShape s, KnownDType d)
            => (Tensor s d -> Builder (Tensor '[] d))
            -> ScanConfig
            -> Kernel s d ScanState ScanInfo
scanSampler logpdf config = Kernel { kernelInit = init, kernelStep = step }
  where
    scaleVal = scanScale config

    init _key pos = do
        ld <- logpdf pos
        return $ ScanState pos ld

    step key state = do
        let pos = scanPosition state
        let ld  = scanLogDens state

        -- Split key for proposal and acceptance
        (key1, key2) <- RNG.splitKey key

        -- Propose: pos' = pos + scale * N(0,1)
        noise <- RNG.rngNormalF32 key1 >>= convert @s @'F32 @d
        scaleT <- constant @s @d scaleVal
        scaledNoise <- tmul noise scaleT
        pos' <- tadd pos scaledNoise

        -- Evaluate proposal
        ld' <- logpdf pos'

        -- Metropolis ratio: log(alpha) = min(0, ld' - ld)
        diff <- tsub ld' ld
        zero <- constant @'[] @d 0.0
        logAlpha <- tminimum diff zero

        -- Uniform draw for acceptance
        u <- RNG.rngUniformF32 key2 >>= convert @'[] @'F32 @d
        logU <- tlog u
        accept <- tlessThan logU logAlpha

        -- Broadcast scalar accept to shape s for element-wise selection
        acceptS <- tbroadcast @'[] @s [] accept

        newPos <- tselect acceptS pos' pos
        newLd  <- tselect accept ld' ld

        infoAcceptProb <- texp logAlpha
        one <- constant @'[] @'I64 1
        let info = Info infoAcceptProb accept one
        return (ScanState newPos newLd, info)
```

What is happening here?

- `RNG.splitKey key` gives us two independent keys.
- `RNG.rngNormalF32 key1` produces a tensor of standard normals with the same shape as the position.
- `convert @s @'F32 @d` converts the `F32` noise to the target dtype `d` (necessary because `rngNormalF32` is hard-coded to `F32` and your kernel may be polymorphic in `d`).
- `tminimum diff zero` computes `min(0, diff)` — the clipped log-acceptance probability.
- `tlessThan logU logAlpha` returns a boolean tensor (really a predicate `0/1` tensor).
- `tbroadcast @'[] @s [] accept` broadcasts the scalar accept flag to the full shape so `tselect` can do element-wise selection.
- `tselect acceptS pos' pos` is the "if-then-else" of tensors: where `acceptS` is true, take from `pos'`; otherwise from `pos`.

---

### 2.4 The compilation and execution pipeline

Once you have a `Kernel`, you cannot run it directly. You must:

1. Build a `Module` from a `Builder` action.
2. Render the module to MLIR text.
3. Compile the MLIR text with PJRT.
4. Create input buffers.
5. Call `execute` with the buffers.
6. Read back the results.

Here is the full pipeline for our custom sampler:

```haskell
import HHLO.IR.AST         (FuncArg(..), TensorType(..))
import HHLO.IR.Builder     (moduleFromBuilder, arg)
import HHLO.IR.Pretty      (render)
import HHLO.Runtime.PJRT.Types (PJRTApi, PJRTClient, PJRTExecutable, PJRTBuffer)
import HHLO.Runtime.Compile    (compileWithOptions, defaultCompileOptions)
import HHLO.Runtime.Execute    (execute)
import HHLO.Runtime.Buffer     (toDeviceF32, fromDeviceF32)
import qualified Data.Vector.Storable as V

-- Step 1: Build a Module
stepModule = moduleFromBuilder @'[2] @'F32 "main"
    [ FuncArg "key" (TensorType [2] UI64)
    , FuncArg "pos" (TensorType [2] F32)
    , FuncArg "ld"  (TensorType [] F32)
    ] $ do
        key <- arg @'[2] @'UI64
        pos <- arg @'[2] @'F32
        ld  <- arg @'[] @'F32
        let kernel = scanSampler myLogPdf (ScanConfig 0.1)
        (state', _info) <- kernelStep kernel (Key key) (ScanState pos ld)
        return (scanPosition state')

-- Step 2: Render to MLIR text
mlirText = render stepModule

-- Step 3: Compile with PJRT
compileModule :: PJRTApi -> PJRTClient -> Module -> IO PJRTExecutable
compileModule api client modl =
    compileWithOptions api client (render modl) defaultCompileOptions

-- Step 4: Create buffers
bufferFromF32 api client dims vals =
    toDeviceF32 api client (V.fromList vals) (map fromIntegral dims)

-- Step 5 & 6: Execute and read back
runOnce api client exe key pos ld = do
    keyBuf <- bufferFromUI64 api client [2] key   -- Word64
    posBuf <- bufferFromF32  api client [2] pos   -- Float
    ldBuf  <- bufferFromF32  api client []  [ld]  -- scalar
    [newPosBuf] <- execute api exe [keyBuf, posBuf, ldBuf]
    result <- fromDeviceF32 api newPosBuf 2
    return (V.toList result)
```

Key observations:

- `moduleFromBuilder` needs the result shape (`'[2]`), dtype (`'F32`), function name (`"main"`), argument list, and the `Builder` body.
- The function name **must** be `"main"` for PJRT to recognise the entry point.
- Argument types in the list must match the types in the `arg` calls inside the body.
- `toDeviceF32` transfers a `Vector Float` from host memory to the device. `fromDeviceF32` does the reverse.
- `execute` takes a list of input buffers and returns a list of output buffers.

---

### 2.5 Buffer management for non-float types

PRNG keys are `Word64`, not `Float`. You need `toDevice` with the correct buffer type:

```haskell
import HHLO.Runtime.PJRT.Types (bufferTypeU64)
import HHLO.Runtime.Buffer     (toDevice, fromDevice)

bufferFromUI64 api client dims vals =
    toDevice api client (V.fromList vals) (map fromIntegral dims) bufferTypeU64

bufferToUI64 api buf n = V.toList <$> fromDevice api buf n
```

The `Common` module in `examples/` already exports `bufferFromUI64` and `bufferToUI64` so you rarely need to write this yourself.

---

### 2.6 The host loop

The compiled `kernelStep` performs one transition. To get a chain, you run it many times on the host:

```haskell
runChain api client stepExe ldExe seed pos0 = do
    -- Evaluate initial log-density
    posBuf0 <- bufferFromF32 api client [2] pos0
    [ldBuf0] <- execute api ldExe [posBuf0]
    [ld0] <- bufferToF32 api ldBuf0 1

    loop seed (0 :: Int) pos0 ld0 100 []
  where
    loop _ _ _ _ 0 acc = return (reverse acc)
    loop seed step pos ld n acc = do
        let key = [seed, fromIntegral step]
        keyBuf <- bufferFromUI64 api client [2] key
        posBuf <- bufferFromF32  api client [2] pos
        ldBuf  <- bufferFromF32  api client []  [ld]
        [newPosBuf] <- execute api stepExe [keyBuf, posBuf, ldBuf]
        newPos <- bufferToF32 api newPosBuf 2
        -- Recompute log-density for the next iteration
        [newLdBuf] <- execute api ldExe [newPosBuf]
        [newLd] <- bufferToF32 api newLdBuf 1
        loop seed (step + 1) newPos newLd (n - 1) (newPos : acc)
```

Notice the pattern:

1. Compile `kernelStep` once.
2. Compile `logpdf` once.
3. In the loop: execute step → read back → recompute density → loop.

The log-density must be recomputed on the host because the *next* `kernelStep` needs it as input. This is a consequence of the single-result limitation: if `kernelStep` could return both the new position *and* the new log-density, we would save one device round-trip per iteration.

---

### 2.7 Conditional logic in `Builder`

MCMC algorithms need branching (accept/reject, slice shrinking, etc.). In HHLO this is done with `conditional`:

```haskell
conditional :: Tensor '[] 'PRED              -- predicate scalar
            -> Builder a                      -- true branch
            -> Builder a                      -- false branch
            -> Builder a
```

For example, to compute the absolute value:

```haskell
absValue x = do
    zero <- constant @'[] @'F32 0.0
    pred <- tlessThan x zero
    negX <- tnegate x
    conditional pred (return negX) (return x)
```

The RandomWalk sampler does not need explicit `conditional` because `tselect` (tensor-level if-then-else) is sufficient for element-wise selection. But more complex algorithms (e.g. NUTS with its tree-doubling logic) will need `conditional` and `whileLoop`.

```haskell
whileLoop :: Tensor '[] 'PRED               -- condition
          -> Builder (Tensor '[] 'PRED)      -- body: returns new condition
          -> Builder ()
```

See `HBayesian.HHLO.Loops` for the full API.

---

## Introspection — Making the API More Intuitive

Now that we have seen both levels, let us reflect on the API design.

### What works well

**Shape safety.** The `Tensor s d` type prevents entire classes of runtime errors. You cannot pass a `[3]` tensor to a function expecting `[2]`. This is a genuine advantage over untyped array libraries.

**Composability.** The `Kernel` record is a clean abstraction. Every sampler exposes the same interface, so the host loop is agnostic to the algorithm.

**Purity.** There is no hidden state, no global RNG, no mutable tensors. Every `Builder` action is deterministic given its inputs. This makes samplers easy to test (render to MLIR and diff) and reason about.

**Single-backend discipline.** There is no "fallback interpreter". If it compiles, it runs on PJRT. This eliminates the "works in test, fails on device" class of bugs.

### What feels mechanical

**Buffer management.** Creating buffers, transferring data, and reading results is verbose and error-prone. The host loop in `runChain` is mostly plumbing.

**Single-result limitation.** Because the PJRT CPU plugin only supports functions with a single return value, every `kernelStep` must be wrapped in a module that returns exactly one tensor. This forces the host to recompute log-density after every step, wasting a device round-trip.

**Explicit key splitting.** Every sampler begins with `splitKey`. This is correct but repetitive. A higher-level combinator could hide this.

**No built-in `sampleChain`.** Users must copy the host loop from an example module. There is no library function like:

```haskell
sampleChain :: Kernel s d state info
            -> Int              -- number of iterations
            -> Tensor s d        -- initial position
            -> IO [Tensor s d]   -- samples
```

### Directions for improvement

> **Status as of v0.2:** The following improvements have been implemented:
> - `sampleChain` combinator with `burnIn`, `thin`, `withSeed`, `parallelChains`
> - `HBayesian.Diagnostics` module with acceptance rate, R-hat, and ESS
> - Shallow PPL layer with `param`, `observe`, and distribution primitives
>
> What remains open: deeper PPL features (plate notation, constrained parameters, structured parameter trees), multi-result PJRT functions, and a monadic `Chain` type for composable workflows.

#### 1. A `sampleChain` combinator

Implemented in v0.2 as `HBayesian.Chain.sampleChain`:

```haskell
sampleChain :: CompiledKernel -> [Float] -> ChainConfig -> IO ([[Float]], [Diagnostic])
```

This hides the entire compilation and execution pipeline:
- Compiles modules inside a fresh PJRT context.
- Manages buffers.
- Runs the host loop.
- Applies burn-in and thinning.
- Returns pure Haskell lists plus diagnostics.

Usage:

```haskell
import HBayesian.Chain

ck <- compileSimpleKernel kernel logpdf
(samples, diags) <- sampleChain ck [0.0, 0.0] $
    burnIn 100 $ thin 2 $ defaultChainConfig { ccNumIterations = 1000 }
```

#### 2. Multi-result support

When PJRT CPU supports multi-result functions (or when we target GPU/TPU where this already works), `kernelStep` could return both the new state and auxiliary values in a single execution. This would eliminate the per-iteration log-density recomputation and roughly halve the device round-trips.

#### 3. Monadic chain composition

A small monadic API for chain manipulation would be powerful:

```haskell
chain :: Chain m => m a -> (a -> m b) -> m b
burnIn :: Int -> Chain a -> Chain a
thin :: Int -> Chain a -> Chain a
concatChains :: [Chain a] -> Chain a
```

This would let users express complex workflows declaratively:

```haskell
myExperiment = do
    warmup <- burnIn 1000 $ sampleChain rwKernel 2000 pos0
    let pos1 = last warmup
    sampleChain hmcKernel 5000 pos1
```

#### 4. PPL layer on top

Implemented in v0.2 as `HBayesian.PPL`:

```haskell
import HBayesian.PPL

myModel :: PPL 2 ()
myModel = do
    alpha <- param 0
    beta  <- param 1
    observe "alpha_prior" (normal 0.0 1.0) alpha
    observe "beta_prior"  (normal 0.0 1.0) beta
    forM_ dataset $ \(x, y) -> do
        mu <- liftBuilder $ tadd alpha =<< tmul beta =<< tconstant x
        sigma <- liftBuilder $ tconstant 0.5
        yT <- liftBuilder $ tconstant y
        observe "y" (normalT mu sigma) yT
```

This desugars to the same `logPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)` that we write manually today.

What remains for future work: `plate` notation, constrained parameter transformations (simplex, positive), and structured parameter trees (pytrees).

#### 5. Automatic differentiation

Phase 5 will add source-to-source AD on StableHLO. When this lands, HMC and MALA will no longer require the user to provide `Gradient`. The API will become:

```haskell
hmcAuto :: (Tensor s d -> Builder (Tensor '[] d))
        -> HMCConfig
        -> Kernel s d (HMCState s d) (Info s d)
hmcAuto logpdf = hmc logpdf (autoGrad logpdf)
```

This preserves backward compatibility (you can still pass an analytical gradient) while making the common case effortless.

### Conclusion

HBayesian v0.2 adds higher-level conveniences — `sampleChain`, chain combinators, diagnostics, and a shallow PPL layer — on top of the solid v0.1 foundation without compromising the core design. The `Kernel` type remains the right abstraction for algorithm designers, while end users can now write models in the PPL and sample with one-line `sampleChain` calls.
