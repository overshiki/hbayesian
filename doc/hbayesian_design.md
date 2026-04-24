# HBayesian: A Design Study for Composable Bayesian Inference in Haskell

> **Executive Summary.** This document analyzes the design of `hbayesian`, a Haskell library that replicates the composable sampling philosophy of BlackJAX using `hhlo`—a Haskell EDSL for StableHLO/XLA—as its sole inference backend. Following the precedent of BlackJAX, which builds exclusively on JAX without a separate "NumPy prototype backend," HBayesian does not entertain a native Haskell fallback. It is HHLO-only.
>
> With the release of **HHLO 0.2.0.0**, two of our three critical infrastructure gaps have been closed by upstream: random number generation (via `rngBitGenerator` with Threefry) and multi-value control flow (via `whileLoop2`, `whileLoopN`, `conditional2`, and `conditionalN`). The remaining hard problem—automatic differentiation—is **consciously deferred to a late stage**. However, this does **not** mean that gradient-based algorithms are deferred. A sampling algorithm is a function that takes a log-density and its gradient as inputs and produces samples as output; it does not care where the gradient comes from. Consequently, HBayesian will support **user-provided manual gradients** from day one, enabling gradient-based samplers (HMC, NUTS, MALA) to ship in the initial release. Auto-grad is merely a convenience layer that will automate gradient computation later, eliminating the need for hand-derived gradients. This phased approach allows the library to ship useful, hardware-accelerated samplers immediately while the AD infrastructure matures in parallel.

---

## 1. Introduction and Motivation

The functional programming paradigm and Bayesian inference share a deep structural affinity. Both are concerned with composing primitive operations according to rigorous mathematical laws to construct complex, correct-by-construction systems. Yet the dominant tooling for Bayesian computation—Stan, PyMC, NumPyro, and BlackJAX—resides in imperative or tracing-based languages (C++, Python). Haskell, despite its mathematical pedigree, lacks a mature, hardware-accelerated library for general-purpose Bayesian inference.

The emergence of **HHLO** changes this landscape. HHLO is a Haskell library that generates **StableHLO MLIR**, the portable intermediate representation of the OpenXLA ecosystem, and executes it on CPU or GPU via the PJRT plugin interface. It provides:
- Type-safe tensor operations with compile-time shape checking (phantom types).
- Direct compilation to XLA without Python or C++ tracing overhead.
- Native multi-GPU execution via `executeReplicas`.
- Pure functional semantics that align with the stateless kernel abstraction used by BlackJAX.

This document asks: *Can we build a BlackJAX-equivalent in Haskell, using HHLO as the numerical backend?* And if so, *what architecture enables algorithms to be written in a unified pattern atop HHLO?*

**A note on architectural discipline.** BlackJAX does not provide a NumPy backend for prototyping. It is JAX-only. Its authors rightly recognized that supporting two backends fractures the codebase, duplicates effort, and creates a temptation to defer hard problems. We adopt the same discipline. HBayesian is **HHLO-only**. Every algorithm, every diagnostic, every utility is built on and compiled through HHLO. There is no escape hatch to pure Haskell vectors or matrices.

Our answer is a **two-tier architecture**:
1. **Algorithm Layer:** Sampling kernels written against a thin abstraction over HHLO.
2. **Backend Layer:** HHLO itself, augmented with the missing capabilities that Bayesian inference demands.

---

## 2. Why Haskell? Why HHLO?

### 2.1 Haskell's Natural Fit for BlackJAX's Design

BlackJAX's core abstraction is the **stateless transition kernel**:

```python
new_state, info = kernel(rng_key, state)
```

In Python, this is a discipline; in Haskell, it is the default. Haskell's purity, referential transparency, and explicit effect tracking (via monads) make the kernel abstraction not merely possible but idiomatic. Key advantages include:

- **No hidden mutable state:** Unlike Python objects, Haskell records and functions cannot surreptitiously modify global state. This eliminates an entire class of bugs in parallel chain execution.
- **Type safety:** Shape errors in tensor operations can be caught at compile time using GHC's type-level naturals and type families—exactly what HHLO already provides via phantom types.
- **Composable control flow:** Higher-order functions, folds, and traversals allow elegant expression of adaptation loops, SMC particle updates, and VI gradient steps.
- **Determinism by default:** The explicit PRNG key pattern used by JAX maps directly to Haskell's explicit state passing or the `State` monad.

### 2.2 HHLO as an XLA Frontend

JAX achieves performance by tracing Python functions to JAXPR, transforming them (JIT, grad, vmap), lowering to XLA HLO, and compiling via the XLA compiler. HHLO short-circuits this pipeline: it generates StableHLO MLIR directly from Haskell and compiles via PJRT. The benefits are:

- **No Python GIL or tracing overhead.**
- **Compile-time shape checking** via phantom types (`Tensor '[2,3] 'F32`).
- **CPU/GPU/TPU portability** through PJRT plugins.
- **Garbage-collected device buffers** via `ForeignPtr` finalizers.
- **Async execution** and **multi-GPU replication** already implemented.

---

## 3. The BlackJAX Model: Functional Requirements

To replicate BlackJAX, we must satisfy the following functional requirements through HHLO:

| Requirement | BlackJAX Implementation | HHLO 0.2.0.0 Status |
|-------------|------------------------|---------------------|
| **Tensor DSL** | JAX NumPy API | ✅ Available (`HHLO.EDSL.Ops`) |
| **Functional RNG** | `jax.random` (Threefry/Philox PRNG) | ⚠️ Partial (`rngBitGenerator` exists; key-based distributions must be built on top) |
| **Multi-value Control Flow** | `jax.lax.scan/while/cond` with pytrees | ⚠️ Partial (`whileLoop2`, `whileLoopN`, `conditional2`, `conditionalN` exist) |
| **JIT Compilation** | `jax.jit` | ✅ Available (`compile` via PJRT) |
| **Vectorization** | `jax.vmap` | ⚠️ Manual batching via leading batch dims |
| **Automatic Differentiation** | `jax.grad` | ❌ Not available (deferred) |
| **Kernels** | Pure Python closures | Haskell closures within the `Builder` monad |
| **Diagnostics** | ArviZ integration | Native Haskell over host-resident results |

---

## 4. Design Philosophy: HHLO-Only

BlackJAX's authors made a crucial decision: they did not support a NumPy or SciPy backend for prototyping. Every operation goes through JAX and XLA. This decision:
- **Eliminates semantic drift:** There is only one behavior for `matmul`, one memory layout, one NaN propagation rule.
- **Forces hard problems to the surface:** If AD is broken, it gets fixed. If RNG is missing, it gets added. There is no comfortable fallback to obscure the pain.
- **Unifies the mental model:** Users and developers think in one abstraction.

We adopt this philosophy wholesale. HBayesian does not have a `Native` backend, a `hmatrix` backend, or a `vector` backend. The only tensor type is the HHLO tensor. The only execution path is PJRT compilation. The only gradient mechanism is the one we build for HHLO—but **we explicitly do not block early releases on its availability**.

---

## 5. The Bayesian Inference DSL (BIDSL)

Rather than a generic "Operation Abstraction Layer" that abstracts over multiple backends, HBayesian defines a **Bayesian Inference DSL** (BIDSL): a module hierarchy that wraps HHLO with the specific vocabulary of probabilistic computation.

### 5.1 Structure

```
HBaysian.HHLO.Ops          -- Re-exports and extends HHLO.EDSL.Ops
HBaysian.HHLO.RNG          -- Functional PRNG: splitKey, normal, uniform, etc.
HBaysian.HHLO.AD           -- Automatic differentiation: grad, jvp, vjp (STUB)
HBaysian.HHLO.Loops        -- Tuple-aware while, scan, cond
HBaysian.HHLO.Compile      -- Compilation and execution helpers
```

Algorithms import only `HBaysian.HHLO.*`, not `HHLO` directly. This gives us:
- **Stability:** If HHLO's internal API changes, we adapt in one place.
- **Semantics:** We can attach Bayesian-specific documentation and invariants to operations.
- **Extensions:** We add missing ops (RNG transformations, AD) without modifying the upstream HHLO repository.

### 5.2 Tensor Type

We reuse HHLO's phantom-typed tensor directly:

```haskell
import HHLO.EDSL.Ops (Tensor)
import HHLO.Core.Types (Shape, DType, F32, F64, I64)

-- A position vector in R^10
type Position = Tensor '[10] 'F64

-- A mass matrix in R^{10x10}
type MassMatrix = Tensor '[10, 10] 'F64

-- A scalar log-density
type LogDensity = Tensor '[] 'F64
```

All algorithms are polymorphic in shape and dtype but monomorphic in backend: the backend is always HHLO.

### 5.3 The Kernel Type

```haskell
module HBayesian.Core where

import HBayesian.HHLO.Ops
import HBayesian.HHLO.RNG

-- | A functional PRNG key.
newtype Key = Key { unKey :: HHLO.Tensor '[2] 'UI64 }

-- | Transition kernel for a specific model shape and dtype.
data Kernel (s :: Shape) (d :: DType) = Kernel
  { kernelInit :: Key -> Tensor s d -> Builder (State s d)
  , kernelStep :: Key -> State s d -> Builder (State s d, Info s d)
  }

data State (s :: Shape) (d :: DType) = State
  { statePosition  :: Tensor s d
  , stateLogDensity :: Tensor '[] d
  }

data Info (s :: Shape) (d :: DType) = Info
  { infoAcceptProb :: Tensor '[] d
  , infoAccepted   :: Tensor '[] 'Bool
  , infoNumSteps   :: Tensor '[] 'I64
  }
```

This is isomorphic to BlackJAX's `SamplingAlgorithm` but typed at the kind level. Note that gradient-dependent fields (momentum, gradient) are omitted from the base `State` type because they are irrelevant to gradient-free samplers.

---

## 6. Infrastructure Status After HHLO 0.2.0.0

### 6.1 Random Number Generation — MOSTLY CLOSED

HHLO 0.2.0.0 added three RNG primitives:

**`rngUniform`** — `stablehlo.rng` with `UNIFORM` distribution
```haskell
rngUniform :: KnownShape s
           => Tensor '[] 'F32 -> Tensor '[] 'F32 -> Builder (Tensor s 'F32)
```

**`rngNormal`** — `stablehlo.rng` with `NORMAL` distribution
```haskell
rngNormal :: KnownShape s => Builder (Tensor s 'F32)
```

**`rngBitGenerator`** — `stablehlo.rng_bit_generator` with **Threefry** algorithm
```haskell
rngBitGenerator :: KnownShape s
                => Tensor '[2] 'UI64
                -> Builder (Tensor '[2] 'UI64, Tensor s 'UI64)
```

**The catch:** `rngUniform` and `rngNormal` are **not key-based**. They use `stablehlo.rng` which accepts no explicit state operand—its behavior is non-deterministic or internally seeded. For a BlackJAX-style **functional, splittable, deterministic PRNG**, we must build on **`rngBitGenerator`** and convert random bits to `F32`/`F64` ourselves.

**Our wrapper design:**

```haskell
module HBayesian.HHLO.RNG where

import HHLO.EDSL.Ops
import HHLO.IR.Builder

newtype Key = Key { unKey :: Tensor '[2] 'UI64 }

splitKey :: Key -> Builder (Key, Key)
splitKey (Key k) = do
  counter1 <- constant @'[] @'UI64 0
  counter2 <- constant @'[] @'UI64 1
  (k1, _) <- rngBitGenerator @'[2] k counter1
  (k2, _) <- rngBitGenerator @'[2] k counter2
  return (Key k1, Key k2)

-- | Generate uniform [0,1) F32 values from a Threefry key.
rngUniformF32 :: KnownShape s => Key -> Builder (Tensor s 'F32)
rngUniformF32 (Key k) = do
  (k', bits) <- rngBitGenerator k
  -- Convert UI64 bits to F32 in [0,1) via bit-casting or division by 2^64
  ...
```

This is feasible because `UI64` is fully supported in HHLO, and `convert` exists for dtype transformations.

### 6.2 Multi-Value Control Flow — PARTIALLY CLOSED

HHLO 0.2.0.0 added:

**`whileLoop2`** — while loop carrying **two tensors of different shapes/dtypes**
```haskell
whileLoop2 :: (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
           => Tensor s1 d1 -> Tensor s2 d2
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tuple2 s1 d1 s2 d2))
           -> Builder (Tuple2 s1 d1 s2 d2)
```

**`whileLoopN`** — while loop carrying **N homogeneous tensors** (same shape & dtype)
```haskell
whileLoopN :: (KnownShape s, KnownDType d)
           => [Tensor s d] -> ... -> Builder [Tensor s d]
```

**`conditional2` / `conditionalN`** — if-then-else returning 2 or N tensors.

**The catch:** For complex MCMC kernels with 4+ mixed state components (e.g., position, momentum, scalar log-density, scalar divergence flag), `whileLoop2` is insufficient and `whileLoopN` requires homogeneous types. In practice, we can:
- **Nest `whileLoop2` calls** for simple cases.
- **Pack scalar state into rank-1 tensors** and use `whileLoopN`.
- **Extend the BIDSL** with `whileLoop3`, `whileLoop4`, etc., which are mechanical boilerplate given the existing infrastructure (`emitOpRegionsN`, `runBlockBuilder`).

For gradient-free samplers (which have simpler state), `whileLoop2` is often sufficient.

### 6.3 Automatic Differentiation — DEFERRED, BUT NOT A BLOCKER

HHLO 0.2.0.0 does not include any automatic differentiation support. We **consciously defer AD to a late stage**—it is a large engineering project (see Appendix A for the full design).

**Crucially, AD is not a blocker for gradient-based samplers.** A gradient-based kernel does not require auto-grad; it merely requires a gradient. If the user can provide `grad_logpdf` by hand, the kernel can execute immediately through HHLO. This is exactly how Stan works: the user writes a model in the Stan language, and the Stan compiler generates gradients automatically. Before the compiler exists, one could still use Stan-style samplers if one wrote the gradient manually.

**Our interface design:**

```haskell
-- | A gradient function provided by the user.
type Gradient s d = Tensor s d -> Builder (Tensor s d)

-- | HMC with a user-supplied gradient.
hmc :: (KnownShape s, KnownDType d)
    => (Tensor s d -> Builder (Tensor '[] d))  -- ^ log-density
    -> Gradient s d                             -- ^ user-provided gradient
    -> HMCConfig
    -> Kernel s d
```

**What gets deferred:**
- The `grad` combinator that automatically derives `Gradient s d` from a log-density function.
- Algorithms that *intrinsically* require differentiating through their own internal computations (e.g., reparameterization gradients for VI, Stein gradients for SVGD). These genuinely need auto-grad and are deferred.

**What ships immediately:**
- HMC, NUTS, MALA, GHMC, Barker, and Laplace-based samplers, all accepting user-provided gradients.
- Random-walk MH and Elliptical Slice (gradient-free) as before.

---

## 7. Algorithm Catalog: Availability by Phase

### Phase 1–2: Gradient-Free Algorithms (No AD Required)

These algorithms require only the log-density function, not its gradient. They can ship as soon as the RNG and loop infrastructure is wrapped.

| Algorithm | Module | Needs Auto-Grad? | Needs User Gradient? | Needs Multi-Value Loops? | Notes |
|-----------|--------|------------------|---------------------|-------------------------|-------|
| **Random-Walk Metropolis-Hastings** | `MCMC.RandomWalk` | No | No | No | Baseline sampler. |
| **Elliptical Slice Sampling** | `MCMC.EllipticalSlice` | No | No | Yes (2-value) | Perfect fit for `whileLoop2`. |
| **Adaptive Random Walk** | `MCMC.AdaptiveRW` | No | No | No | Mass matrix adaptation via empirical covariance. |
| **HMC** | `MCMC.HMC` | No | **Yes** | Yes | Ships immediately with user gradients. |
| **MALA** | `MCMC.MALA` | No | **Yes** | No | Ships immediately with user gradients. |
| **NUTS** | `MCMC.NUTS` | No | **Yes** | Yes | Ships immediately with user gradients. |
| **GHMC** | `MCMC.GHMC` | No | **Yes** | Yes | Ships immediately with user gradients. |
| **Barker** | `MCMC.Barker` | No | **Yes** | No | Ships immediately with user gradients. |
| **Tempered SMC** | `SMC.Tempered` | No | No | Yes (particle loop) | Uses `whileLoopN` over particles. |
| **SMC with RW Mutation** | `SMC.FromMCMC` | No | No | Yes | Combines SMC with gradient-free kernels. |

### Phase 3: Algorithms Intrinsically Requiring Auto-Grad

These algorithms require differentiating through internal computations that the user cannot easily provide by hand.

| Algorithm            | Module            | Needs Auto-Grad? | Notes                                                                                                    |
| -------------------- | ----------------- | ---------------- | -------------------------------------------------------------------------------------------------------- |
| **Mean-Field VI**    | `VI.MeanField`    | **Yes**          | ELBO gradients via reparameterization through the variational distribution.                              |
| **Full-Rank VI**     | `VI.FullRank`     | **Yes**          | Same; gradients w.r.t. Cholesky factors are tedious to derive manually.                                  |
| **Pathfinder**       | `VI.Pathfinder`   | **Yes**          | L-BFGS objective gradients; feasible manually for small models, deferred for generality.                 |
| **SVGD**             | `VI.SVGD`         | **Yes**          | Kernelized Stein discrepancy requires automatic kernel gradients.                                        |
| **SGLD**             | `SGMCMC.SGLD`     | **Yes**          | Minibatch gradients of log-posterior; user can provide manually, but auto-grad is the value proposition. |
| **SGHMC**            | `SGMCMC.SGHMC`    | **Yes**          | Same as SGLD.                                                                                            |
| **SGNHT**            | `SGMCMC.SGNHT`    | **Yes**          | Same as SGLD.                                                                                            |
| **RMHMC**            | `MCMC.RMHMC`      | **Yes**          | Requires Hessians or Fisher information; too complex for manual derivation in most models.               |
| **Laplace Samplers** | `MCMC.LaplaceHMC` | **Yes**          | Require Hessians for Laplace approximation.                                                              |

---

## 8. Algorithm Implementation Architecture

With the BIDSL in place, algorithms are written once, compiled through HHLO, and executed on any PJRT-supported device.

### 8.1 The Log-Density Function

The user provides a log-density as a Haskell function:

```haskell
logDensity :: forall d. (KnownDType d)
           => Tensor '[10] d -> Builder (Tensor '[] d)
logDensity x = do
  xSq <- tmul x x
  prior <- tnegate (tsum @'[10] @'[] [0] xSq)
  return prior
```

This function constructs an HHLO computation graph. It is not executed until compiled via PJRT.

### 8.2 Building a Gradient-Free Kernel: Random-Walk MH

```haskell
module HBayesian.MCMC.RandomWalk where

import HBayesian.Core
import HBayesian.HHLO.Ops
import HBayesian.HHLO.RNG

randomWalk :: forall s d.
              (KnownShape s, KnownDType d)
           => (Tensor s d -> Builder (Tensor '[] d))  -- ^ log-density
           -> Tensor '[] 'F64                        -- ^ proposal scale
           -> Kernel s d
randomWalk logpdf scale = Kernel {..}
  where
    kernelInit key pos = do
      ld <- logpdf pos
      return $ State pos ld

    kernelStep key state = do
      let (key1, key2) = splitKey key
      let pos = statePosition state
      let ld = stateLogDensity state
      -- Proposal: pos' ~ N(pos, scale^2 * I)
      noise <- rngNormal key1
      scaledNoise <- tmul noise (tbroadcast scale)
      pos' <- tadd pos scaledNoise
      ld' <- logpdf pos'
      -- Metropolis acceptance
      logAccept <- tsub ld' ld
      u <- rngUniform key2
      let accept = tlessThan (tlog u) logAccept
      newPos <- tselect accept pos' pos
      newLd  <- tselect accept ld' ld
      let info = Info (texp logAccept) accept (constant @'[] @'I64 1)
      return (State newPos newLd, info)
```

Notice that every operation lives in the `Builder` monad. The kernel constructs a computational graph; when compiled via PJRT, the entire MH step—proposal generation, log-density evaluation, and acceptance test—executes as a single fused XLA kernel.

### 8.3 Elliptical Slice Sampling

Elliptical slice sampling is an ideal early algorithm because it:
- Requires no gradients.
- Uses a simple `whileLoop2` (position and auxiliary Gaussian).
- Has no Metropolis correction.

```haskell
module HBayesian.MCMC.EllipticalSlice where

import HBayesian.Core
import HBayesian.HHLO.Ops
import HBayesian.HHLO.RNG
import HBayesian.HHLO.Loops

ellipticalSlice :: forall s d.
                   (KnownShape s, KnownDType d)
                => (Tensor s d -> Builder (Tensor '[] d))  -- ^ log-density
                -> Kernel s d
ellipticalSlice logpdf = Kernel {..}
  where
    kernelInit key pos = do
      ld <- logpdf pos
      return $ State pos ld

    kernelStep key state = do
      let (key1, key2) = splitKey key
      pos <- return $ statePosition state
      -- Sample auxiliary nu ~ N(0, I)
      nu <- rngNormal key1
      -- Sample initial angle
      u <- rngUniform key2
      theta <- tmul u (tconstant (2 * pi))
      -- Define the ellipse point: pos * cos(theta) + nu * sin(theta)
      let ellipse th = do
            c <- tcos th
            s <- tsin th
            pc <- tmul pos c
            ns <- tmul nu s
            tadd pc ns
      -- Bracket expansion loop using whileLoop2
      (theta', _, accepted) <- whileLoop2 theta (tconstant True)
        (\th _ -> do
          -- shrinking logic
          ...)
      pos' <- ellipse theta'
      ld' <- logpdf pos'
      let info = Info (tconstant 1.0) (tconstant True) (tconstant 1)
      return (State pos' ld', info)
```

### 8.4 Compilation and Execution

The outer inference loop is written in Haskell but calls compiled PJRT executables:

```haskell
module HBayesian.Compile where

import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

-- | Compile a kernel step to a PJRT executable.
compileKernelStep :: Kernel s d -> IO PJRTLoadedExecutable
compileKernelStep kernel = do
  let moduleText = renderModule $ buildModule $ do
        key <- arg @'[2] @'UI64
        pos <- arg @s @d
        state0 <- kernelInit kernel (Key key) pos
        (state1, info) <- kernelStep kernel (Key key) state0
        returnTuple2 (statePosition state1) (infoAcceptProb info)
  compileWithOptions api client moduleText defaultCompileOptions

-- | Run one step.
runStep :: PJRTLoadedExecutable -> Key -> HostVector -> IO (HostVector, Double)
runStep exec key pos = do
  keyBuf <- hostToBuffer key
  posBuf <- hostToBuffer pos
  [outBuf] <- execute api exec [keyBuf, posBuf]
  outHost <- bufferToHost outBuf
  return (outHost, 0.0)
```

For performance, the entire chain should be compiled into a single XLA computation using a compiled `while` loop. This eliminates host-device synchronization at every step.

---

## 9. Random Number Generation in Depth

### 9.1 The JAX PRNG Design

JAX uses a **counter-based PRNG** (Threefry2x32 or Philox) where the key is an explicit value, not hidden global state. Functions take a `PRNGKey` and return a new one. Keys can be split to produce statistically independent subkeys:

```python
key, subkey = jax.random.split(key)
```

This is essential for JIT compilation and parallelism: the compiler can see all data dependencies, and `vmap` over chains simply maps over different subkeys.

### 9.2 Mapping to HBayesian

HHLO 0.2.0.0 provides `rngBitGenerator` with the **Threefry** algorithm, which is the same algorithm JAX uses. The state is a `Tensor '[2] 'UI64`. We wrap this into a `Key` newtype and provide `splitKey`:

```haskell
module HBayesian.HHLO.RNG where

import HHLO.EDSL.Ops
import HHLO.IR.Builder

newtype Key = Key { unKey :: Tensor '[2] 'UI64 }

splitKey :: Key -> Builder (Key, Key)
splitKey (Key k) = do
  c1 <- constant @'[] @'UI64 0
  c2 <- constant @'[] @'UI64 1
  (k1, _) <- rngBitGenerator @'[2] k c1
  (k2, _) <- rngBitGenerator @'[2] k c2
  return (Key k1, Key k2)

-- | Uniform [0,1) F32 via Threefry + bit-to-float conversion.
rngUniformF32 :: KnownShape s => Key -> Builder (Tensor s 'F32)
rngUniformF32 (Key k) = do
  (k', bitsUI64) <- rngBitGenerator k
  -- bit-cast or scale UI64 to F32 in [0,1)
  bitsF32 <- convert bitsUI64
  maxVal <- tconstant (fromIntegral (maxBound :: Word64))
  scaled <- tdiv bitsF32 maxVal
  return scaled
```

**Note on `rngNormal` and `rngUniform` from HHLO:** The built-in `rngNormal` and `rngUniform` in `HHLO.EDSL.Ops` use `stablehlo.rng`, which lacks an explicit state operand. For deterministic, splittable keys, we ignore these and build our own distributions atop `rngBitGenerator`.

### 9.3 Parallel Chains

With explicit keys, running 100 chains is trivial. We create a batch of keys and add a batch dimension to the position tensor:

```haskell
runChains :: forall s d. (KnownShape s, KnownDType d)
          => Int -> Kernel s d -> Key -> Tensor (Batch ': s) d -> Builder (Tensor (Batch ': s) d)
runChains n kernel key0 positions = do
  let keys = splitKeyN n key0
  batchedStep <- vmapKernel kernel
  scan (batchedStep keys) positions
```

In XLA, this becomes a single computation with a leading batch dimension, automatically parallelized across GPU cores.

---

## 10. Type System and Shape Safety

One of HBayesian's advantages over BlackJAX is **compile-time shape checking**. JAX checks shapes at trace time (runtime from the user's perspective). HHLO checks many shapes at Haskell compile time via phantom types.

### 10.1 Type-Level Shapes and DTypes

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

import GHC.TypeNats
import Data.Proxy

type Shape = [Nat]

data DType = F32 | F64 | I32 | I64 | Bool

class KnownShape (s :: Shape) where
  shapeVal :: Proxy s -> [Int]

instance KnownShape '[] where shapeVal _ = []
instance (KnownNat n, KnownShape s) => KnownShape (n ': s) where
  shapeVal _ = fromIntegral (natVal (Proxy @n)) : shapeVal (Proxy @s)
```

This allows operations like `matmul` to enforce shape compatibility at compile time, exactly as HHLO already does.

### 10.2 Existential Shapes for Dynamic Models

Not all models have statically known shapes. A hierarchical model might have a parameter vector whose length depends on the number of groups in the dataset. For these cases, we use existentials:

```haskell
module HBayesian.Dynamic where

import Data.Some (Some)

data SomeTensor d where
  SomeTensor :: (KnownShape s) => Tensor s d -> SomeTensor d

-- A log-density function that works with any shape
LogDensity d = SomeTensor d -> Builder (Tensor '[] d)
```

This mirrors JAX's dynamic shapes but provides a principled type-safe boundary.

---

## 11. Diagnostics and Utilities

BlackJAX delegates convergence diagnostics to ArviZ. HBayesian provides native implementations:

- **Effective Sample Size (ESS):** Spectral density-based ESS (FFT or autocovariance methods).
- **Potential Scale Reduction ($\hat{R}$):** Gelman-Rubin split-$\hat{R}$ across multiple chains.
- **Divergence Tracking:** Count and fraction of divergent transitions (for NUTS, when it arrives).

These are pure functions over `Vector Double` and do not require backend abstraction. They operate on the **host-resident results** returned by PJRT after sampling.

---

## 12. Implementation Roadmap

### Phase 1: BIDSL Foundation (Weeks 1–4)
- Define `HBayesian.Core` kernel types and `HBayesian.HHLO.Ops` re-exports.
- Implement `Key` newtype and `splitKey` atop `rngBitGenerator`.
- Implement `rngUniformF32`, `rngNormalF32`, `rngBernoulli` via bit-to-float conversion.
- Wrap `whileLoop2` and `whileLoopN` for Bayesian state patterns.
- Add `whileLoop3`, `whileLoop4` boilerplate for richer state (mechanical, given `emitOpRegionsN`).
- Write golden tests verifying rendered MLIR for RNG and loop wrappers.

### Phase 2: Core MCMC Samplers (Weeks 5–10)
- Implement **Random-Walk Metropolis-Hastings**.
- Implement **Elliptical Slice Sampling**.
- Implement **Adaptive Random Walk** with online covariance adaptation.
- Implement **HMC** and **MALA** accepting user-provided gradients.
- Implement host-side inference loop + PJRT execution pipeline.
- Write end-to-end tests comparing HHLO CPU results against reference Python implementations.
- Write GPU smoke tests.

### Phase 3: NUTS and Advanced MCMC (Weeks 11–16)
- Implement **NUTS** with user-provided gradients.
- Implement **GHMC** and **Barker** proposals.
- Implement **Tempered SMC** with systematic resampling.
- Implement **SMC with HMC/MALA mutation kernels** (user gradients).
- Implement multi-chain parallelism via leading batch dimensions.
- Implement ESS and $\hat{R}$ diagnostics.
- Benchmark against BlackJAX/NumPyro on CPU and GPU.

### Phase 4: Automatic Differentiation (Months 5–9)
- Implement reverse-mode AD on the HHLO AST (see Appendix A).
- Start with VJP rules for the ~20 ops needed by HMC.
- Build `grad` combinator integrating with `Builder`.
- Add memory optimization (rematerialization).
- Provide `grad`-based convenience wrappers: `hmcAuto`, `nutsAuto`, etc.

### Phase 5: Variational Inference (Months 9–12)
- Implement **Mean-Field Gaussian VI**.
- Implement **Full-Rank Gaussian VI**.
- Implement **Pathfinder**.
- Implement **SVGD**.

### Phase 6: Advanced Gradient Methods (Months 12–14)
- Implement **RMHMC** (requires Hessians via auto-grad).
- Implement **Laplace-based samplers**.
- Implement **SGLD**, **SGHMC**, **SGNHT** with minibatching.

### Phase 7: Polish (Ongoing)
- Haddock documentation, tutorials, and examples.
- Criterion benchmarks and CI.
- Hackage release.

---

## 13. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **AD implementation is harder than estimated** | Medium | High | Already mitigated by deferral. Gradient-based samplers still ship with user-provided gradients. |
| **XLA dynamic shapes too limited for NUTS** | Medium | High | Use fixed-cap padded trajectories (standard practice); defer NUTS to Phase 5 anyway. |
| **HHLO maintenance stalls or API breaks** | Low | High | BIDSL isolates algorithms from builder changes; we maintain our own RNG/loop wrappers. |
| **Performance inferior to JAX** | Medium | Medium | Acceptable for gradient-free phase; optimize fusion via XLA HLO dumps in later phases. |
| **Type-level shape programming is too complex for users** | Medium | Medium | Provide `SomeTensor` escape hatch; invest in documentation. |
| **Threefry RNG bit-to-float conversion has numerical issues** | Low | Medium | Use high-precision conversion; validate against JAX output. |

---

## 14. Comparison with the Ecosystem

| Dimension | BlackJAX | NumPyro | Stan | HBayesian (proposed) |
|-----------|----------|---------|------|---------------------|
| Language | Python | Python | Stan/C++ | Haskell |
| Backend | JAX/XLA | JAX/XLA | C++ Autodiff | HHLO/XLA |
| PPL | No | Yes | Yes | No |
| Modularity | High | Medium | Low | Very High (type-class parameterized) |
| GPU | Native | Native | Limited | Native (via PJRT) |
| Compile-time safety | None | None | Some | Extensive (shape/dtype checking) |
| AD | `jax.grad` | `jax.grad` | Auto | **Deferred** (manual interim, auto late) |
| Initial samplers | NUTS, HMC, RW | NUTS, HMC | NUTS, HMC | **RW-MH, Elliptical Slice, HMC, MALA, NUTS** (user gradients + gradient-free) |
| Functional purity | Discipline | Discipline | N/A | Enforced by language |
| Fallback backend | None | None | None | None (HHLO-only) |

HBayesian's unique trajectory is **shipping useful, hardware-accelerated samplers immediately** while AD matures in the background. Where BlackJAX launched with NUTS, HBayesian launches with Elliptical Slice and Random-Walk MH—algorithms that are slower per sample but require no gradients, run on GPU via XLA, and prove the end-to-end pipeline.

---

## 15. Conclusion

Building `hbayesian` as a Haskell counterpart to BlackJAX, backed exclusively by HHLO/XLA, is an ambitious but coherent project. The release of **HHLO 0.2.0.0** has materially changed the feasibility landscape: random number generation and multi-value control flow—two of our three hard problems—are now supported by upstream. We need only wrap them with a Bayesian-specific API.

By **deferring automatic differentiation to a late stage**, we remove the single largest blocker to an initial release. Gradient-free samplers (random-walk Metropolis-Hastings, elliptical slice sampling, and SMC) are theoretically sound, widely used, and fully implementable today. They will serve as the proof of concept for the HHLO-only architecture, the BIDSL design, and the PJRT execution pipeline.

When AD eventually arrives—implemented as source-to-source transformation on the HHLO AST—it will plug into the existing BIDSL without breaking changes. The gradient-based algorithms (HMC, NUTS, VI, SGMCMC) will then elevate HBayesian from a useful research tool to a direct competitor to BlackJAX.

The discipline is HHLO-only. The path is gradient-free first. The destination is a type-safe, hardware-accelerated, composable Bayesian inference library in Haskell.

---

## References

1. Cabezas, A., Corenflos, A., Lao, J., & Louf, R. (2024). BlackJAX: Composable Bayesian inference in JAX. *arXiv:2402.10797*.
2. Bradbury, J., Frostig, R., Hawkins, P., et al. (2018). JAX: Composable transformations of Python+NumPy programs. *GitHub: google/jax*.
3. HHLO Contributors. (2025). *HHLO: Haskell library for StableHLO*. GitHub: overshiki/hhlo. Version 0.2.0.0.
4. OpenXLA Contributors. (2023). *StableHLO specification*. GitHub: openxla/stablehlo.
5. Gelman, A., Carlin, J. B., Stern, H. S., et al. (2013). *Bayesian Data Analysis*. CRC Press.
6. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.
7. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler. *JMLR*, 15(1), 1593–1623.
8. Murray, I., Adams, R. P., & MacKay, D. J. (2010). Elliptical slice sampling. *AISTATS*.
9. Zhang, L., et al. (2022). Pathfinder: Parallel quasi-Newton variational inference. *JMLR*, 23(1), 1–49.
10. Dau, H. D., & Chopin, N. (2022). Waste-free Sequential Monte Carlo. *JRSS-B*, 84(1), 114–148.

---

## Appendix A: Source-to-Source Automatic Differentiation on the HHLO AST

> This appendix records the detailed design for reverse-mode automatic differentiation on the HHLO AST. While AD is **deferred to a late stage** of the HBayesian roadmap, its architecture has already been analyzed and is preserved here so that implementation can proceed without redesign when the project reaches Phase 5.

### A.1 The Gap

HHLO 0.2.0.0 provides no automatic differentiation. JAX's `jax.grad` works by tracing Python functions to JAXPR, applying the JAXPR autodiff transform, and lowering to XLA HLO. HHLO generates StableHLO directly, bypassing JAXPR entirely. There is no `grad` function in the HHLO API.

### A.2 Solution: Source-to-Source AD on the HHLO AST

HHLO constructs an MLIR AST in `HHLO.IR.AST` before pretty-printing. We traverse this AST and apply reverse-mode automatic differentiation by implementing the chain rule for each StableHLO op. For every primitive op `y = f(x)`, we define its backward rule:

```
  x_bar += vjp(f, x, y_bar)
```

This is precisely how JAX's XLA backend works: each HLO op has a predefined JVP/VJP translation. The StableHLO specification documents the semantics of ~100 ops; for each, we implement a Haskell function that constructs the gradient computation graph.

### A.3 Why This Is Feasible

1. **Closed op set:** StableHLO defines a fixed set of primitives. Unlike Python, where users can define arbitrary differentiable functions, we only need VJP rules for the primitives used by sampling algorithms. The active set is small: approximately 20–30 ops (`add`, `subtract`, `multiply`, `divide`, `matmul`, `dot_general`, `sum`, `reduce`, `broadcast_in_dim`, `reshape`, `transpose`, `slice`, `concatenate`, `dynamic_slice`, `gather`, `scatter`, `exp`, `log`, `sin`, `cos`, `power`, `sqrt`, `rsqrt`, `tanh`, `erf`, `select`, `compare`, `convert`, `negate`, `abs`, `minimum`, `maximum`, `constant`).

2. **Pure AST:** The HHLO AST is a pure, traversable data structure. There is no Python dynamism, no `__getattr__`, no runtime code generation. We pattern-match on `Operation` values and rewrite them.

3. **Known math:** XLA's autodiff already exists in C++. We are not inventing new mathematics; we are porting known VJP rules into Haskell.

### A.4 The Trace-Capture Problem

The `Builder` monad in HHLO is stateful: it appends operations to a mutable list (inside `State`) and returns `ValueId` handles. To differentiate a function, we need to capture the sequence of operations it emits.

**Approach:** Run the forward function in an isolated `Builder` sub-session, capture the emitted operation list, then run the backward pass by appending new operations to the same (or a derived) builder context.

```haskell
module HBayesian.HHLO.AD where

import HHLO.IR.AST
import HHLO.IR.Builder
import qualified Data.Map.Strict as M

-- | A cotangent map: from ValueId to its accumulated cotangent ValueId.
type CotangentMap = M.Map ValueId ValueId

-- | Reverse-mode AD transform.
-- Takes a computation represented as a Builder action from input tensor to
-- scalar output, and returns a Builder action that computes the gradient.
grad :: forall s d.
        (KnownShape s, KnownDType d)
     => (Tensor s d -> Builder (Tensor '[] d))
     -> (Tensor s d -> Builder (Tensor s d))
grad f = \x -> do
  -- Run forward pass and capture the trace
  (y, trace) <- captureTrace (f x)
  -- Initialize cotangent of output to 1.0
  yBar <- constant @'[] @d 1.0
  let initCotangents = M.singleton (tensorValue y) (tensorValue yBar)
  -- Backpropagate in reverse topological order
  cotMap <- backpropTrace trace initCotangents
  -- Extract gradient w.r.t. input x
  case M.lookup (tensorValue x) cotMap of
    Just xBar -> return (Tensor xBar)
    Nothing   -> error "grad: input has no cotangent (unused in forward trace)"
```

`captureTrace` runs the inner `Builder` action in a fresh scope, records every emitted operation, and returns both the result and the ordered trace.

### A.5 Backpropagation Through the Trace

The `backpropTrace` function processes operations in **reverse order** (from output to input). For each operation, it looks up the cotangents of its results and applies the corresponding VJP rule to compute cotangents for its operands.

```haskell
backpropTrace :: [Operation] -> CotangentMap -> Builder CotangentMap
backpropTrace ops cotMap = foldr backpropOp (return cotMap) (reverse ops)

backpropOp :: Operation -> Builder CotangentMap -> Builder CotangentMap
backpropOp op mcot = do
  cotMap <- mcot
  case opName op of
    "stablehlo.add"      -> vjpAdd op cotMap
    "stablehlo.multiply" -> vjpMultiply op cotMap
    "stablehlo.dot"      -> vjpDot op cotMap
    "stablehlo.exp"      -> vjpExp op cotMap
    "stablehlo.log"      -> vjpLog op cotMap
    -- ... etc. for all 20+ ops
    _ -> error ("backpropOp: no VJP rule for " ++ show (opName op))
```

### A.6 Example VJP Rules

#### A.6.1 Addition: `z = add(x, y)`

```haskell
vjpAdd :: Operation -> CotangentMap -> Builder CotangentMap
vjpAdd op cotMap = do
  let [xVid, yVid] = opOperands op
      [zVid]       = opResults op
  case M.lookup zVid cotMap of
    Nothing -> return cotMap  -- z is not used downstream
    Just zBar -> do
      -- x_bar += z_bar, y_bar += z_bar
      let update v = M.alter (\mb -> case mb of
            Nothing -> Just zBar
            Just b  -> Just (emitAdd b zBar))
      return (update xVid (update yVid cotMap))
```

#### A.6.2 Multiplication: `z = multiply(x, y)`

```haskell
vjpMultiply :: Operation -> CotangentMap -> Builder CotangentMap
vjpMultiply op cotMap = do
  let [xVid, yVid] = opOperands op
      [zVid]       = opResults op
  case M.lookup zVid cotMap of
    Nothing -> return cotMap
    Just zBar -> do
      xBar <- emitMul zBar yVid
      yBar <- emitMul zBar xVid
      let update v b = M.alter (\mb -> case mb of
            Nothing -> Just b
            Just b0 -> Just (emitAdd b0 b))
      return (update xVid xBar (update yVid yBar cotMap))
```

#### A.6.3 Matrix Multiplication: `z = dot(x, y)`

```haskell
vjpDot :: Operation -> CotangentMap -> Builder CotangentMap
vjpDot op cotMap = do
  let [xVid, yVid] = opOperands op
      [zVid]       = opResults op
  case M.lookup zVid cotMap of
    Nothing -> return cotMap
    Just zBar -> do
      -- x_bar += z_bar @ y^T
      xBar <- emitDot zBar yVid (transposeAttrs yVid)
      -- y_bar += x^T @ z_bar
      yBar <- emitDot xVid zBar (transposeAttrs xVid)
      ...
```

#### A.6.4 Element-wise Functions: `y = exp(x)`

```haskell
vjpExp :: Operation -> CotangentMap -> Builder CotangentMap
vjpExp op cotMap = do
  let [xVid] = opOperands op
      [yVid] = opResults op
  case M.lookup yVid cotMap of
    Nothing -> return cotMap
    Just yBar -> do
      -- x_bar += y_bar * y
      xBar <- emitMul yBar yVid
      return (M.alter (Just . maybe xBar (emitAdd xBar)) xVid cotMap)
```

#### A.6.5 Reduction: `y = reduce_sum(x, dims)`

```haskell
vjpReduceSum :: Operation -> CotangentMap -> Builder CotangentMap
vjpReduceSum op cotMap = do
  let [xVid] = opOperands op
      [yVid] = opResults op
      dims   = extractReduceDims op
  case M.lookup yVid cotMap of
    Nothing -> return cotMap
    Just yBar -> do
      -- x_bar += broadcast(y_bar, x_shape, dims)
      xBar <- emitBroadcastInDim yVid (inputShape op) dims
      return (M.alter (Just . maybe xBar (emitAdd xBar)) xVid cotMap)
```

### A.7 Broadcasting and Shape Transformations

Broadcasting, reshape, transpose, slice, and concatenate require careful tracking of shape metadata. The VJP rules must invert these transformations:

- **`broadcast_in_dim`**: The backward pass is `reduce_sum` over the broadcasted dimensions.
- **`reshape`**: The backward pass is `reshape` back to the original shape.
- **`transpose`**: The backward pass is `transpose` with the inverse permutation.
- **`slice`**: The backward pass is `pad` with zeros in the sliced-out regions.
- **`concatenate`**: The backward pass is `slice` for each input along the concat dimension.

Because HHLO carries shape information at the type level, many of these backward shapes can be computed statically. For dynamic shapes (e.g., `dynamic_slice`), we use the same dynamic indices in the backward pass.

### A.8 Memory Optimization

#### A.8.1 Rematerialization

For cheap element-wise ops (`exp`, `log`, `sin`, `negate`), we do not store the forward value. Instead, we recompute it in the backward pass:

```haskell
vjpExpRemat :: Operation -> CotangentMap -> Builder CotangentMap
vjpExpRemat op cotMap = do
  let [xVid] = opOperands op
      [yVid] = opResults op
  case M.lookup yVid cotMap of
    Nothing -> return cotMap
    Just yBar -> do
      -- Recompute y = exp(x) rather than loading it from memory
      yVidRecomputed <- emitExp xVid
      xBar <- emitMul yBar yVidRecomputed
      ...
```

#### A.8.2 Checkpointing

For expensive ops (large `matmul`, `convolution`), we save the output. For cheap ops, we recompute. This tradeoff is controlled by a `CheckpointPolicy` data type:

```haskell
data CheckpointPolicy = RematerializeCheap | CheckpointExpensive
```

### A.9 Integration with the BIDSL

When AD is implemented, the `HBayesian.HHLO.AD` module exports a single combinator:

```haskell
module HBayesian.HHLO.AD where

grad :: (KnownShape s, KnownDType d)
     => (Tensor s d -> Builder (Tensor '[] d))
     -> (Tensor s d -> Builder (Tensor s d))
```

Algorithm modules import this combinator. During the deferral period, the module exports a placeholder:

```haskell
grad :: a
grad = error "AD is not yet implemented. Use manual gradients or gradient-free samplers."
```

This ensures that algorithm code can be written against the `grad` interface today and will work automatically when AD lands.

### A.10 References for AD Implementation

1. Pearlmutter, B. A., & Siskind, J. M. (2008). Reverse-mode AD in a functional framework: Lambda the ultimate backpropagator. *ACM TOPLAS*, 30(2), 1–36.
2. Wang, F., Wu, X., Essertel, G., Decker, J., & Rompf, T. (2018). Demystifying differentiable programming: Shift/reset the penultimate backpropagator. *arXiv:1803.10228*.
3. JAX Autodiff Documentation. *github.com/google/jax/blob/main/docs/autodidax.md* — A pedagogical implementation of JAX's autodiff from scratch.
4. StableHLO Op Semantics. *openxla.org/stablehlo/spec* — Definitive reference for forward and backward behavior of each op.

---

*Document updated following the release of HHLO 0.2.0.0. The HHLO-only discipline is adopted by conscious choice, following the precedent of BlackJAX. Automatic differentiation is deferred; gradient-free samplers lead the launch.*
