# HBayesian

Composable Bayesian inference in Haskell on StableHLO / XLA.

HBayesian provides MCMC samplers that compile to [StableHLO](https://github.com/openxla/stablehlo) and execute via PJRT (CPU, GPU, or TPU). It includes RandomWalk MH, Elliptical Slice, HMC, MALA, and NUTS — along with chain combinators, diagnostics, and a shallow probabilistic programming layer.

## What makes HBayesian different

### Write Haskell, run on XLA

HBayesian is built on a simple idea: write ordinary Haskell functions that construct tensor computation graphs, then compile those graphs to StableHLO and execute them via PJRT.

You write ordinary Haskell functions that construct tensor computation graphs. Those graphs compile to self-contained StableHLO modules and execute on PJRT — the same runtime that powers JAX, TensorFlow, and PyTorch XLA. The result is that your sampler runs as a single XLA program, with the entire MCMC step (random number generation, proposal, gradient evaluation, acceptance) happening on device.

```haskell
-- This is not interpreted. It builds a StableHLO graph.
myLogPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
myLogPdf theta = do
    alpha <- tslice1 @2 @'F32 theta 0
    beta  <- tslice1 @2 @'F32 theta 1
    -- ... graph construction ...
```

### Compile-time shape safety

Tensors carry their shape and dtype in the type:

```haskell
x :: Tensor '[3] 'F32   -- 1-D float vector of length 3
y :: Tensor '[] 'F32    -- scalar float
```

Mismatches are caught at compile time. You cannot pass a `[3]` parameter vector to a sampler expecting `[2]` — the error appears in GHC, not at runtime on the device. This eliminates an entire class of "shape mismatch" bugs that are common in frameworks with dynamic typing.

### One language, end to end

Model definition, sampler design, and execution control are all Haskell. There is no Python/C++ boundary to cross, no foreign-function interface to debug, and no "works in the interpreter but fails on GPU" surprises. If your code type-checks, it compiles to StableHLO. If it compiles to StableHLO, it runs on PJRT.

### Composable samplers

Every sampler exposes the same `Kernel` interface:

```haskell
data Kernel s d state info = Kernel
  { kernelInit :: Key -> Tensor s d -> Builder (state s d)
  , kernelStep :: Key -> state s d -> Builder (state s d, info s d)
  }
```

This means the host loop is agnostic to the algorithm inside. You can swap RandomWalk for HMC without changing your execution code. The library provides five kernels out of the box:

| Sampler | Needs gradient? | Best for |
|---------|-----------------|----------|
| **RandomWalk** MH | No | Low dimensions, simple models |
| **EllipticalSlice** | No | Gaussian priors, no tuning needed |
| **HMC** | Yes | High dimensions, informed proposals |
| **MALA** | Yes | Cheap gradient steps |
| **NUTS** | Yes | Adaptive trajectory, no tuning of `L` |

### Chain combinators

Sampling is configured through ordinary function composition:

```haskell
sampleChain ck [0.0, 0.0] $
    burnIn 500 $ thin 2 $ withSeed 42 $ defaultChainConfig
        { ccNumIterations = 2000 }
```

`parallelChains` runs multiple independent chains with distinct seeds — essential for Gelman-Rubin convergence diagnostics.

### Diagnostics

`HBayesian.Diagnostics` operates on the `Diagnostic` records collected by `sampleChain`:

- **Acceptance rate** — fraction of accepted proposals
- **Gelman-Rubin R-hat** — across parallel chains
- **Effective sample size (ESS)** — autocorrelation-adjusted

### PPL layer

Write models as generative stories instead of manual log-densities:

```haskell
myModel :: PPL 2 ()
myModel = do
    alpha <- param 0
    beta  <- param 1
    observe "alpha_prior" (normal 0.0 1.0) alpha
    observe "beta_prior"  (normal 0.0 1.0) beta
```

`runPPL` desugars the story into a standard log-posterior function.

## Quick start

### Install the PJRT plugin

```bash
./scripts/pjrt_script.sh
```

This downloads `libpjrt_cpu.so` into `deps/pjrt/`. To use a custom plugin:

```bash
export HBAYESIAN_PJRT_PLUGIN=/path/to/libpjrt_cpu.so
```

### Build

```bash
cabal build
```

### Run a demo

```bash
# Run all 5 basic examples
cabal run hbayesian-examples -- --execute

# Run the HMC goodness-of-fit regression test
cabal run correlated-gaussian-hmc
```

## Usage

### Basic sampling

```haskell
import HBayesian.Chain
import HBayesian.MCMC.RandomWalk
import HBayesian.Diagnostics (acceptanceRate)
import qualified LinearRegressionRandomWalk as Ex

main :: IO ()
main = do
    let kernel = randomWalk Ex.linearRegLogPdf (RWConfig 0.1)
        ck     = compileSimpleKernel kernel Ex.linearRegLogPdf
    (samples, diags) <- sampleChain ck [0.0, 0.0] $
        burnIn 100 $ thin 2 $ defaultChainConfig { ccNumIterations = 1000 }
    print (head samples)
    putStrLn $ "Acceptance rate: " ++ show (acceptanceRate diags)
```

### HMC with gradients

```haskell
import HBayesian.Chain
import HBayesian.MCMC.HMC

main :: IO ()
main = do
    let config = HMCConfig { hmcStepSize = 0.1, hmcNumLeapfrogSteps = 10 }
        kernel = hmc myLogPdf myGradient config
        ck     = compileHMC kernel myLogPdf myGradient
    (samples, diags) <- sampleChain ck (replicate 5 0.0) $
        burnIn 500 $ defaultChainConfig { ccNumIterations = 2000 }
    print (head samples)
```

### NUTS (No-U-Turn Sampler)

```haskell
import HBayesian.Chain
import HBayesian.MCMC.NUTS

main :: IO ()
main = do
    let config = NUTSConfig { nutsStepSize = 0.05
                            , nutsMaxDepth = 10
                            , nutsDeltaMax = 1000.0 }
        kernel = nuts myLogPdf myGradient config
        ck     = compileNUTS kernel myLogPdf myGradient
    (samples, diags) <- sampleChain ck (replicate 10 0.0) $
        burnIn 500 $ defaultChainConfig { ccNumIterations = 1000 }
    print (head samples)
```

NUTS eliminates the need to tune the number of leapfrog steps (`L`). It builds a binary tree of leapfrog trajectories and stops automatically when the trajectory begins to double back (a "U-turn"). The entire tree construction happens inside nested XLA `whileLoop` ops.

### Parallel chains

```haskell
results <- parallelChains 4 (map (+ 0.5)) ck pos0 config
let chains = map fst results
```

### PPL layer

```haskell
import HBayesian.PPL

myModel :: PPL 2 ()
myModel = do
    alpha <- param 0
    beta  <- param 1
    observe "alpha_prior" (normal 0.0 1.0) alpha
    observe "beta_prior"  (normal 0.0 1.0) beta

logpdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
logpdf = runPPL myModel
```

## Examples

| Example | Sampler | What it shows | Run |
|---------|---------|---------------|-----|
| `LinearRegressionRandomWalk` | RandomWalk MH | Bayesian linear regression, 2-D params | `cabal run hbayesian-examples -- --execute` |
| `GaussianProcessEllipticalSlice` | Elliptical Slice | GP regression with Gaussian prior | `cabal run hbayesian-examples -- --execute` |
| `LogisticRegressionHMC` | HMC | Logistic regression with user-provided gradient | `cabal run hbayesian-examples -- --execute` |
| `BivariateGaussianMALA` | MALA | 2-D correlated Gaussian target | `cabal run hbayesian-examples -- --execute` |
| `CorrelatedGaussianHMC` | HMC | 5-D Gaussian with statistical GoF validation | `cabal run correlated-gaussian-hmc` |
| `CorrelatedGaussianNUTS` | NUTS | Same 5-D Gaussian, using NUTS | `cabal run correlated-gaussian-nuts` |
| `NealFunnel` | HMC + NUTS | Benchmark showing NUTS efficiency advantage | `cabal test --test-options '-p NealFunnel'` |

Each example exposes `makeKernel` and `runChain`. Import them in GHCi to experiment interactively:

```bash
cabal repl hbayesian-examples
```

```haskell
import qualified LinearRegressionRandomWalk as Ex
Ex.runChain
```

### CorrelatedGaussianHMC — GoF regression test

`CorrelatedGaussianHMC` is the most rigorous example in the suite. It targets a 5-D correlated Gaussian with AR(1) covariance (ρ = 0.7) where the precision matrix and gradient are derived analytically. Because the model is exact, any deviation in the samples points directly to a sampler bug.

Running it prints a statistical report with four checks:

- **Marginal moments** — each dimension's mean and variance match N(μᵢ, 1)
- **Kolmogorov–Smirnov** — each marginal passes a KS test against the theoretical CDF
- **Mahalanobis χ²** — mean of (x−μ)ᵀΛ(x−μ) ≈ 5
- **Gelman–Rubin R-hat** — across 4 parallel chains, R-hat < 1.1 for every dimension

```bash
cabal run correlated-gaussian-hmc
```

Output:
```
==================================
  Goodness-of-Fit Report
==================================
Sample count: 2000

Marginal means (expected vs observed):
  dim 0:     1.0 vs  1.025  (diff: 2.54e-2)  PASS
...

Mahalanobis distances:
  expected mean: 5.00
  observed mean: 5.04  PASS
```

### NealFunnel — NUTS efficiency benchmark

`NealFunnel` targets Neal's funnel distribution, the canonical example where NUTS dramatically outperforms fixed-step HMC. The geometry changes drastically across the posterior: at small `y`, the `x` variables are tightly concentrated; at large `y`, they are very diffuse. No single fixed trajectory length works well everywhere.

The experiment compares four configurations on the same 10-D target:

| Configuration | Parameters | Mean ESS |
|---------------|-----------|----------|
| HMC-short | `L = 10` leapfrog steps | ~44 |
| HMC-medium | `L = 50` leapfrog steps | ~50 |
| HMC-long | `L = 200` leapfrog steps | ~63 |
| **NUTS** | `max_depth = 10` (adaptive) | **~284** |

NUTS achieves roughly **4.5× higher mean ESS** than the best fixed-step HMC, because it avoids wasting computation on U-turns while still exploring effectively. The entire binary tree construction — direction sampling, leapfrog integration, slice checking, U-turn detection, and candidate selection — happens inside compiled XLA `whileLoop` ops.

Run the benchmark via the test suite:

```bash
cabal test --test-options '-p NealFunnel'
```

Or use the module directly in GHCi:

```bash
cabal repl hbayesian-examples
```

```haskell
import NealFunnel
(samples, diags) <- runNUTS
print (head samples)
```

## Tests

```bash
cabal test
```

39 tests covering Core, HHLO primitives, RNG, loops, all 5 samplers (including NUTS), chain combinators, PPL, the CorrelatedGaussian GoF suite, and the NealFunnel efficiency benchmark.

## Tutorial

A comprehensive two-level tutorial lives in `tutorial/TUTORIAL.md`:

- **Level 1** — Using inference algorithms (for end users)
- **Level 2** — Designing inference algorithms (for algorithm authors)

## Project structure

```
hbayesian/
|-- src/HBayesian/
|   |-- Chain.hs              -- Chain combinators
|   |-- Diagnostics.hs        -- MCMC diagnostics
|   |-- PPL.hs                -- Probabilistic programming layer
|   |-- MCMC/                 -- Samplers (RandomWalk, EllipticalSlice, HMC, MALA, NUTS)
|   |-- HHLO/                 -- StableHLO primitives, RNG, loops, PJRT helpers
|-- examples/                 -- Example suite + CLI entry points
|-- test/                     -- Test suite
|-- tutorial/TUTORIAL.md
```

## Status

- **Phase 1** (Foundation): ✅ Core abstractions, RNG, loops, compilation
- **Phase 2** (Samplers): ✅ 5 samplers + practical examples
- **Phase 3** (Usability): ✅ Chain combinators, PPL, diagnostics, GoF validation
- **Phase 4+**: Adaptation, auto-diff — see `doc/` for design documents

## License

MIT
