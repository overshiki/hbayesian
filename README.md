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

## Setup

### 1. Install the PJRT plugin

HBayesian uses the same plugin-download strategy as HHLO. Run the bundled script once:

```bash
./scripts/pjrt_script.sh
```

This downloads `libpjrt_cpu.so` (and optionally GPU plugins if detected) into `deps/pjrt/`. The directory is gitignored.

**Override:** If you already have a PJRT plugin installed elsewhere, you can either set the environment variable:

```bash
export HBAYESIAN_PJRT_PLUGIN=/path/to/libpjrt_cpu.so
```

or pass it as a flag (see Tier B below).

### 2. Build

```bash
cabal build
```

## Usage

### Tier A — Render MLIR (works without PJRT)

Each example can render its `kernelStep` to StableHLO MLIR text for inspection:

```bash
cabal run hbayesian-examples -- --render
```

`--render` is the default when no flags are given.

### Tier B — Execute on PJRT

Run the chains on the device:

```bash
cabal run hbayesian-examples -- --execute
```

With a custom plugin path:

```bash
cabal run hbayesian-examples -- --execute --pjrt-plugin /path/to/libpjrt_cpu.so
```

This compiles each example's kernels to PJRT executables, runs a short chain on the device, and prints the sampled positions. See all available flags with `--help`.

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

A `Kernel` is a pure specification — it describes *what* one MCMC step does, not *how* to run it. The step is a `Builder` action that manipulates tensors. To actually sample, you compile it to StableHLO and execute via PJRT.

### From kernel to executable

The full pipeline looks like this:

1. **Compile the log-posterior** (and gradient, if required) into separate PJRT executables so the host can evaluate them to initialise the chain.
2. **Compile `kernelStep`** into a PJRT executable. Because the PJRT CPU plugin only supports single-result functions, wrap the step so it returns exactly one tensor (usually the new position).
3. **Create device buffers** for the PRNG key, position, log-density, and any auxiliary state.
4. **Run a host loop**: execute the step executable, read back the new position, recompute log-density/gradient for the next iteration, and repeat.

The examples in `examples/` demonstrate the complete wiring. Here is the conceptual shape (see `LinearRegressionRandomWalk.hs` for the full implementation):

```haskell
import HBayesian.HHLO.PJRT
import Common

runChain :: IO [[Float]]
runChain = withPJRTCPU $ \api client -> do
    -- 1. Compile log-posterior
    ldExe <- compileModule api client ldModule

    -- 2. Compile kernel step (single-result wrapper)
    stepExe <- compileModule api client stepModule

    -- 3. Create initial buffers
    keyBuf  <- bufferFromUI64 api client [2] [seed, 0]
    posBuf  <- bufferFromF32  api client [2] [0.0, 0.0]
    ...

    -- 4. Host loop
    loop api client stepExe ldExe ... 10 []
```

### Run a chain programmatically

If you just want to see it work, import any example module and call its `runChain`:

```haskell
import HBayesian.HHLO.PJRT
import qualified LinearRegressionRandomWalk as Ex

main :: IO ()
main = do
  samples <- Ex.runChain
  print (head samples)
```

## Examples

Practical examples live in `examples/` and demonstrate every Phase 2 sampler on a real statistical problem:

| Example | Sampler | Model | Mode |
|---------|---------|-------|------|
| `LinearRegressionRandomWalk.hs` | RandomWalk MH | Bayesian linear regression (2-D params) | Render + Execute |
| `GaussianProcessEllipticalSlice.hs` | Elliptical Slice | GP regression with identity prior covariance | Render + Execute |
| `LogisticRegressionHMC.hs` | HMC | Bayesian logistic regression with user-provided gradient | Render + Execute |
| `BivariateGaussianMALA.hs` | MALA | 2-D correlated Gaussian target | Render + Execute |

Each example exposes:
- `makeKernel` — factory for the sampler kernel
- `renderStepMlir` — rendered MLIR text of a single `kernelStep`
- `runChain` — actual PJRT execution that returns a list of posterior samples

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
|-- scripts/
|   |-- pjrt_script.sh                 -- Download PJRT plugins
|-- src/
|   |-- HBayesian/
|   |   |-- Core.hs                    -- Kernel, State, Info, Key, Gradient
|   |   |-- HHLO/
|   |   |   |-- Ops.hs                 -- Primitives, comparisons, convenience aliases
|   |   |   |-- RNG.hs                 -- Threefry PRNG, uniform/normal/bernoulli
|   |   |   |-- Loops.hs               -- whileLoop, conditional
|   |   |   |-- Compile.hs             -- renderBuilder, compileModule
|   |   |   |-- PJRT.hs                -- Plugin discovery + execution helpers
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
- **Phase 2** (MCMC Samplers): ✅ Complete — RandomWalk, EllipticalSlice, HMC, MALA + 4 practical examples with PJRT execution
- **Phase 3+**: NUTS, adaptation, variational inference, auto-diff, and more — see `doc/` for design documents

## License

MIT
