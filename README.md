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

You can pass an analytical gradient directly. Auto-differentiation (source-to-source AD on StableHLO) is planned for Phase 5, but the API will not change when it arrives.

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

## Usage

### Define a model

A model is just a log-posterior function:

```haskell
myLogPdf :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
myLogPdf theta = do
  alpha <- tslice1 @2 @'F32 theta 0
  beta  <- tslice1 @2 @'F32 theta 1
  -- ... likelihood + prior ...
```

### Build a kernel

```haskell
import HBayesian.MCMC.RandomWalk

kernel :: SimpleKernel '[2] 'F32
kernel = randomWalk myLogPdf (RWConfig 0.1)
```

### Render the step to MLIR (Tier A — works today)

```haskell
import HBayesian.HHLO.Compile

mlirText :: Text
mlirText = renderBuilder (kernelStep kernel key state) module0
```

### Run a chain (Tier B — requires PJRT)

```haskell
import HBayesian.InferenceLoop

samples <- sampleChain defaultInferenceConfig kernel key0 initialPosition
```

`sampleChain` is currently a stub awaiting PJRT CPU plugin integration. When enabled, the same `kernel` works unchanged.

## Examples

Practical examples live in `examples/` and demonstrate every Phase 2 sampler on a real statistical problem:

| Example | Sampler | Model |
|---------|---------|-------|
| `LinearRegressionRandomWalk.hs` | RandomWalk MH | Bayesian linear regression (2-D params) |
| `GaussianProcessEllipticalSlice.hs` | Elliptical Slice | GP regression with identity prior covariance |
| `LogisticRegressionHMC.hs` | HMC | Bayesian logistic regression with user-provided gradient |
| `BivariateGaussianMALA.hs` | MALA | 2-D correlated Gaussian target |

Run all examples and print their StableHLO MLIR:

```bash
cabal run hbayesian-examples
```

Each example exposes:
- `makeKernel` — factory for the sampler kernel
- `renderStepMlir` — rendered MLIR text of a single `kernelStep`
- `runChain` — stub for end-to-end sampling (Tier B)

## Tests

```bash
cabal test
```

The test suite includes:
- **Core** — type sanity checks for `Key`, `State`, `Info`, `Kernel`
- **HHLO.Ops** — golden rendering tests for missing primitives (`sqrt`, `sin`, `cos`, `pow`, comparisons)
- **HHLO.RNG** — smoke tests for `splitKey`, `rngUniformF32`, `rngNormalF32`
- **HHLO.Loops** — rendering tests for `whileLoop` and `conditional`
- **MCMC** — MLIR smoke tests for all four samplers
- **Examples** — MLIR smoke tests for all four practical examples

## Project structure

```
hbayesian/
|-- src/
|   |-- HBayesian/
|   |   |-- Core.hs                    -- Kernel, State, Info, Key, Gradient
|   |   |-- HHLO/
|   |   |   |-- Ops.hs                 -- Primitives, comparisons, convenience aliases
|   |   |   |-- RNG.hs                 -- Threefry PRNG, uniform/normal/bernoulli
|   |   |   |-- Loops.hs               -- whileLoop, conditional
|   |   |   |-- Compile.hs             -- renderBuilder, compileModule
|   |   |-- InferenceLoop.hs           -- sampleChain (stub)
|   |   |-- MCMC/
|   |       |-- RandomWalk.hs
|   |       |-- EllipticalSlice.hs
|   |       |-- HMC.hs
|   |       |-- MALA.hs
|-- examples/                          -- Practical example suite
|   |-- Common.hs
|   |-- LinearRegressionRandomWalk.hs
|   |-- GaussianProcessEllipticalSlice.hs
|   |-- LogisticRegressionHMC.hs
|   |-- BivariateGaussianMALA.hs
|-- test/                              -- Test suite
|   |-- Test/
|   |   |-- Core.hs
|   |   |-- HHLO/Ops.hs
|   |   |-- HHLO/RNG.hs
|   |   |-- HHLO/Loops.hs
|   |   |-- MCMC.hs
|   |   |-- Examples.hs
```

## Status

- **Phase 1** (BIDSL Foundation): ✅ Complete — Core abstractions, RNG, loops, compilation utilities
- **Phase 2** (MCMC Samplers): ✅ Complete — RandomWalk, EllipticalSlice, HMC, MALA + 4 practical examples
- **Phase 3+**: NUTS, adaptation, variational inference, auto-diff, and more — see `doc/` for design documents

## License

MIT
