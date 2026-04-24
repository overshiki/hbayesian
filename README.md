# HBayesian

> Composable Bayesian inference in Haskell on StableHLO / XLA

HBayesian is a library of MCMC samplers that compiles inference kernels to [StableHLO](https://github.com/openxla/stablehlo) and executes them via PJRT (CPU, GPU, or TPU). It is inspired by [BlackJAX](https://github.com/blackjax-devs/blackjax) but built exclusively on top of the [HHLO](https://hackage.haskell.org/package/hhlo) Haskell EDSL — there is no native Haskell fallback. Every tensor operation becomes an XLA graph.

## Design

### HHLO-only backend

All samplers are written as pure `Builder` actions in the HHLO EDSL. A single MCMC step — including random number generation, proposal, gradient evaluation, and acceptance — compiles to a self-contained StableHLO module. This module is then lowered to PJRT and executed on the device.

### User-provided gradients from day one

The `Gradient` type is a simple function:

```haskell
type Gradient s d = Tensor s d -> Builder (Tensor s d)
```

You can pass an analytical gradient directly. 

### Kernel abstraction

Every sampler exposes the same `Kernel` record:

```haskell
data Kernel s d state info = Kernel
  { kernelInit :: Key -> Tensor s d -> Builder (state s d)
  , kernelStep :: Key -> state s d -> Builder (state s d, info s d)
  }
```

This makes samplers composable: the host-side loop (`sampleChain`) is agnostic to the specific transition kernel it is driving.

## Samplers

| Sampler | Module | Gradient required? |
|---------|--------|-------------------|
| Random-Walk Metropolis-Hastings | `HBayesian.MCMC.RandomWalk` | No |
| Elliptical Slice Sampling | `HBayesian.MCMC.EllipticalSlice` | No |
| Hamiltonian Monte Carlo (HMC) | `HBayesian.MCMC.HMC` | **Yes** |
| MALA | `HBayesian.MCMC.MALA` | **Yes** (thin wrapper over HMC with 1 leapfrog step) |

## Setup

### 1. Install the PJRT plugin

HBayesian uses the same plugin-download strategy as HHLO. Run the bundled script once:

```bash
./scripts/pjrt_script.sh
```

This downloads `libpjrt_cpu.so` (and optionally GPU plugins if detected) into `deps/pjrt/`. The directory is gitignored.

**Override:** If you already have a PJRT plugin installed elsewhere, set the environment variable:

```bash
export HBAYESIAN_PJRT_PLUGIN=/path/to/libpjrt_cpu.so
```

### 2. Build

```bash
cabal build
```

## Tutorial

A comprehensive two-level tutorial lives in `tutorial/TUTORIAL.md`:

- **Level 1** — Using inference algorithms (for end users)
- **Level 2** — Designing inference algorithms (for algorithm authors)


## Usage

### Run a chain

The simplest way is through the chain combinators in `HBayesian.Chain`:

```haskell
import HBayesian.Chain
import HBayesian.MCMC.RandomWalk
import qualified LinearRegressionRandomWalk as Ex

main :: IO ()
main = do
    let kernel = randomWalk Ex.linearRegLogPdf (RWConfig 0.1)
    let ck = compileSimpleKernel kernel Ex.linearRegLogPdf
    (samples, diags) <- sampleChain ck [0.0, 0.0] $
        burnIn 100 $ thin 2 $ defaultChainConfig { ccNumIterations = 1000 }
    print (head samples)
    putStrLn $ "Acceptance rate: " ++ show (acceptanceRate diags)
```

Available combinators: `compileSimpleKernel`, `compileHMC`, `sampleChain`, `burnIn`, `thin`, `withSeed`, `parallelChains`.

Diagnostics (acceptance rate, R-hat, ESS) live in `HBayesian.Diagnostics`.

### Define a model

A model is just a log-posterior function:

```haskell
myLogPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
myLogPdf theta = do
    alpha <- tslice1 @2 @'F32 theta 0
    beta  <- tslice1 @2 @'F32 theta 1
    -- ... likelihood + prior ...
```

Or use the PPL layer to write it as a generative story:

```haskell
import HBayesian.PPL

myModel :: PPL 2 ()
myModel = do
    alpha <- param 0
    beta  <- param 1
    observe "alpha_prior" (normal 0.0 1.0) alpha
    observe "beta_prior"  (normal 0.0 1.0) beta
    -- likelihood observations ...

logpdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
logpdf = runPPL myModel
```

### Build a kernel

```haskell
import HBayesian.MCMC.RandomWalk

kernel :: SimpleKernel '[2] 'F32
kernel = randomWalk myLogPdf (RWConfig 0.1)
```

A `Kernel` is a pure specification — it describes *what* one MCMC step does, not *how* to run it. The step is a `Builder` action that manipulates tensors. To actually sample, compile it with `compileSimpleKernel` or `compileHMC` and run via `sampleChain`.

### Inspect MLIR (works without PJRT)

Render a sampler's `kernelStep` to StableHLO MLIR text:

```bash
cabal run hbayesian-examples -- --render
```

## Examples

Practical examples live in `examples/` and demonstrate every sampler on a real statistical problem:

| Example | Sampler | Model | Run |
|---------|---------|-------|-----|
| `LinearRegressionRandomWalk.hs` | RandomWalk MH | Bayesian linear regression (2-D params) | `cabal run hbayesian-examples -- --execute` |
| `GaussianProcessEllipticalSlice.hs` | Elliptical Slice | GP regression with identity prior covariance | `cabal run hbayesian-examples -- --execute` |
| `LogisticRegressionHMC.hs` | HMC | Bayesian logistic regression with user-provided gradient | `cabal run hbayesian-examples -- --execute` |
| `BivariateGaussianMALA.hs` | MALA | 2-D correlated Gaussian target | `cabal run hbayesian-examples -- --execute` |
| `CorrelatedGaussianHMC.hs` | HMC | 5-D AR(1) Gaussian with analytical gradient + GoF tests | `cabal run correlated-gaussian-hmc` |

Each example exposes `makeKernel` (factory for the sampler kernel) and `runChain` (runs a short chain and returns samples).

### Running a single example

The CLI executable `hbayesian-examples` runs **all 4 basic examples in sequence**. To run just one:

**Option 1 — GHCi (quickest)**

```bash
cabal repl hbayesian-examples
```

Then in the REPL:

```haskell
import qualified LinearRegressionRandomWalk as Ex
Ex.runChain
```

**Option 2 — Small standalone script**

Create `RunOne.hs` in the project root:

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

import HBayesian.Diagnostics (acceptanceRate)
import qualified LinearRegressionRandomWalk as Ex

main :: IO ()
main = do
    (samples, diags) <- Ex.runChain
    print (head samples)
    putStrLn $ "Acceptance rate: " ++ show (acceptanceRate diags)
```

Run it:

```bash
cabal exec -- ghc -iexamples -isrc RunOne.hs -o run_one \
  -package-db dist-newstyle/packagedb/ghc-9.6.7
./run_one
```

### CorrelatedGaussianHMC — GoF regression test

`CorrelatedGaussianHMC` is not just another demo — it is a **statistical regression test** for the HMC implementation. It targets a 5-D correlated Gaussian with AR(1) covariance (ρ = 0.7) where the precision matrix Λ and its gradient are derived analytically. Because the model is exact, any deviation in the samples points directly to a sampler bug.

Run it and see the report:

```bash
cabal run correlated-gaussian-hmc
```

The report applies four independent checks to the HMC samples:

- **Marginal moments** — each dimension's mean and variance match N(μᵢ, 1)
- **Kolmogorov–Smirnov** — each marginal passes a KS test against the theoretical CDF
- **Mahalanobis χ²** — mean of (x−μ)ᵀΛ(x−μ) ≈ 5 (degrees of freedom)
- **Gelman–Rubin R-hat** — across 4 parallel chains, R-hat < 1.1 for every dimension

You can also use it programmatically:

```haskell
import qualified CorrelatedGaussianHMC as Ex

main :: IO ()
main = do
    (samples, diags) <- Ex.runChainV2
    Ex.goodnessOfFitReport samples
```

## Tests

```bash
cabal test
```

The test suite includes 34 tests:
- **Core** — type sanity checks for `Key`, `State`, `Info`, `Kernel`
- **HHLO.Ops** — golden rendering tests for missing primitives (`sqrt`, `sin`, `cos`, `pow`, comparisons)
- **HHLO.RNG** — rendering and key-splitting tests for `splitKey`, `rngUniformF32`, `rngNormalF32`
- **HHLO.Loops** — rendering tests for `whileLoop` and `conditional`
- **MCMC** — MLIR smoke tests for all four samplers
- **Examples** — MLIR smoke tests for all practical examples
- **Chain** — `sampleChain` and `parallelChains` execution tests
- **PPL** — PPL-derived model rendering tests
- **CorrelatedGaussian** — rigorous GoF tests for HMC on a 5-D Gaussian (marginal means/variances, KS tests, Mahalanobis distance, R-hat across 4 chains)

## Project structure

```
hbayesian/
|-- scripts/
|   |-- pjrt_script.sh                 -- Download PJRT plugins
|-- src/
|   |-- HBayesian/
|   |   |-- Core.hs                    -- Kernel, State, Info, Key, Gradient
|   |   |-- Chain.hs                   -- Chain combinators: compileSimpleKernel, compileHMC, sampleChain, parallelChains
|   |   |-- Diagnostics.hs             -- Host-side MCMC diagnostics (acceptanceRate, rHat, ess)
|   |   |-- PPL.hs                     -- Shallow probabilistic programming layer
|   |   |-- HHLO/
|   |   |   |-- Ops.hs                 -- Primitives, comparisons, convenience aliases
|   |   |   |-- RNG.hs                 -- Threefry PRNG, uniform/normal/bernoulli
|   |   |   |-- Loops.hs               -- whileLoop, conditional
|   |   |   |-- Compile.hs             -- renderBuilder, compileModule
|   |   |   |-- PJRT.hs                -- Plugin discovery + execution helpers
|   |   |-- MCMC/
|   |   |   |-- RandomWalk.hs
|   |   |   |-- EllipticalSlice.hs
|   |   |   |-- HMC.hs
|   |   |   |-- MALA.hs
|-- examples/                          -- Practical example suite
|   |-- Common.hs
|   |-- Main.hs                        -- CLI entry point for hbayesian-examples
|   |-- CorrelatedGaussianHMCMain.hs   -- CLI entry point for correlated-gaussian-hmc
|   |-- LinearRegressionRandomWalk.hs
|   |-- GaussianProcessEllipticalSlice.hs
|   |-- LogisticRegressionHMC.hs
|   |-- BivariateGaussianMALA.hs
|   |-- CorrelatedGaussianHMC.hs       -- 5-D Gaussian with GoF validation
|-- test/                              -- Test suite
|   |-- Test/
|   |   |-- Core.hs
|   |   |-- HHLO/Ops.hs
|   |   |-- HHLO/RNG.hs
|   |   |-- HHLO/Loops.hs
|   |   |-- MCMC.hs
|   |   |-- Examples.hs
|   |   |-- Chain.hs
|   |   |-- PPL.hs
|   |   |-- CorrelatedGaussian.hs
```

## Status

- **Phase 1** (BIDSL Foundation): ✅ Complete — Core abstractions, RNG, loops, compilation utilities
- **Phase 2** (MCMC Samplers): ✅ Complete — RandomWalk, EllipticalSlice, HMC, MALA + 4 practical examples with PJRT execution
- **Phase 3** (Chain Combinators, PPL, GoF): ✅ Complete — `sampleChain`, `parallelChains`, `burnIn`, `thin`; shallow PPL (`param`, `observe`, `normal`, `uniform`, etc.); rigorous GoF example with 34 passing tests
- **Phase 4+**: NUTS, adaptation, variational inference, auto-diff, and more — see `doc/` for design documents

## License

MIT
