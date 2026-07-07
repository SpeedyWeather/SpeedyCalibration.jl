# SpeedyCalibration.jl — Package Design Plan

## Purpose

Gradient-based parameter calibration for SpeedyWeather.jl using online statistical
gradient estimation (Enzyme AD through a single timestep, averaged over a running
simulation). This package extracts and stabilises the calibration workflow developed
in the master thesis notebooks into a reusable, composable Julia package suitable
for inclusion in the SpeedyWeather GitHub organisation.

---

## Problem Statement

The notebooks in `differentiability/main/point_wise_loss/` contain three interleaved
concerns that make them hard to reuse:

1. **Infrastructure** — `ParamSpec`, sigmoid reparameterisation, path helpers
2. **Loss & gradients** — flux extraction, target/weight tables, Enzyme AD pass
3. **Training loop** — spinup, batch accumulation, optimizer step, history tracking
4. **Analysis** — `DailyMeansCallback`, climate validation runs, plotting

Every new experiment (`5param_osr_temp_train`, `online_train_multi_param`,
`full_trend_birth_training`) copy-pastes and re-versions all four layers, creating
fragmented state (v5, v6, v7, v1, v2, v3) that is hard to reproduce or extend.

The package separates these concerns into stable modules with a clean public API.

---

## Repository Layout

```
SpeedyCalibration.jl/
├── src/
│   ├── SpeedyCalibration.jl   ← module root, re-exports public API
│   ├── param_spec.jl          ← ParamSpec, path helpers, sigmoid reparameterisation
│   ├── loss.jl                ← LossConfig, flux extraction, weighted-MSE computation
│   ├── gradients.jl           ← compute_gradients! (Enzyme AD pass, state save/restore)
│   ├── training.jl            ← TrainingConfig, calibrate! (the main training loop)
│   ├── callbacks.jl           ← DailyMeansCallback, ClimateSnapshotCallback
│   ├── validation.jl          ← build_climate_sim, run_climate_validation
│   └── analysis.jl            ← TrainingResult, plot_training, plot_climate
├── examples/
│   ├── radiation_sw.jl        ← reproduces the 15-param OSR+SRU+SRD result
│   └── trenberth_full.jl      ← reproduces the full 6-flux Trenberth result
├── test/
│   ├── runtests.jl
│   ├── test_param_spec.jl     ← unit tests: ParamSpec, sigmoid, path helpers
│   └── test_smoke.jl          ← integration: 5-batch calibration at trunc=5
├── Project.toml
└── README.md
```

---

## Key Types

### `ParamSpec`

Defines a single trainable parameter. Unchanged from the notebooks.

```julia
struct ParamSpec
    name       :: Symbol
    path       :: Vector{Symbol}          # property path into model, e.g. [:shortwave_radiation, :cloud_albedo]
    bounds     :: Tuple{Float32,Float32}  # physical [lb, ub] — enforced via sigmoid
    initial    :: Union{Nothing,Float32}  # nothing → read from model default
    grad_scale :: Float32                 # per-parameter effective learning-rate multiplier
end
```

Constructor: `ParamSpec(name, path; bounds, initial, grad_scale)` — same as today.

### `LossConfig`

Replaces the hard-coded `TRENBERTH_TARGETS`/`TRENBERTH_WEIGHTS` dictionaries and the
per-function seeding code. Decouples *which fluxes* from *how the AD pass works*.

```julia
struct LossConfig
    flux_keys :: Vector{Symbol}              # e.g. [:osr, :sru, :srd]
    targets   :: Dict{Symbol,Float32}        # W/m² reference values
    weights   :: Dict{Symbol,Float32}        # dimensionless loss weights
end
```

Pre-built configs:

```julia
const OSR_LOSS        = LossConfig([:osr]; targets=Dict(:osr=>101.9f0), weights=Dict(:osr=>1f0))
const OSR_SRU_SRD_LOSS = LossConfig([:osr,:sru,:srd]; ...)
const TRENBERTH_LOSS  = LossConfig([:osr,:sru,:srd,:olr,:lrd,:lru]; ...)   # 6-flux
```

`LossConfig` also owns the mapping from `flux_key` → field path inside
`variables.parameterizations`, so `gradients.jl` can iterate generically instead of
having a separate function per loss variant.

### `TrainingConfig`

Bundles all hyperparameters into one struct so call-sites are not 20-keyword functions.

```julia
Base.@kwdef struct TrainingConfig
    spinup_days         :: Int       = 180
    batch_days          :: Float64   = 3.0
    samples_per_batch   :: Int       = 20
    max_batches         :: Int       = 300
    loss_window_size    :: Int       = 20
    loss_threshold      :: Float32   = 500f0
    patience            :: Int       = 30
    grad_clip           :: Float32   = 5f0
    enable_lr_decay     :: Bool      = true
    lr_decay_factor     :: Float32   = 0.5f0
    lr_plateau_patience :: Int       = 50
    min_lr              :: Float32   = 1f-6
    max_lr_decays       :: Int       = 3
    trunc               :: Int       = 31
    nlayers             :: Int       = 8
    start_date          :: DateTime  = DateTime(2000, 3, 21)
    verbose             :: Bool      = true
end
```

A `quick_test_config()` convenience function returns `TrainingConfig(spinup_days=5,
max_batches=5, trunc=5)` for fast iteration.

### `TrainingResult`

Replaces the bare `(history, final_params, conv_info)` tuple.

```julia
struct TrainingResult
    history      :: Dict{Symbol,Vector}
    final_params :: Dict{Symbol,Float32}
    conv_info    :: NamedTuple
    config       :: TrainingConfig
    loss_config  :: LossConfig
    param_specs  :: Vector{ParamSpec}
end
```

Supports `save(result, path)` / `load(TrainingResult, path)` via JLD2 for
resume-safe long runs.

---

## Core Functions

### `compute_gradients!(variables, model, loss_config, param_specs) → (grads, fluxes, loss)`

Lives in `gradients.jl`. This is the function that touches Enzyme.

- Computes area-weighted Gaussian means for all `flux_keys` in `loss_config`
- Saves full prognostic state (vorticity, divergence, temperature, humidity, pressure,
  ocean, land, variables.grid)
- Builds cotangent seeds per ring: `seed_ij = 2·w·(mean−target)·gw_j / (nlons_j·w_sum)`
- Calls `Enzyme.autodiff(Reverse, timestep!, ...)`
- Restores state
- Returns `(gradients::Vector{Float32}, fluxes::NamedTuple, loss::Float32)`

One function handles all `LossConfig` variants — no more `compute_gradients_osr_sru!`
vs `compute_gradients_trenberth!` duplication.

### `enzyme_warmup(; trunc=5, nlayers=3)`

Lives in `gradients.jl`. Runs one AD pass on a tiny model to force Enzyme compilation.
Should be called once per session before `calibrate!`.

### `calibrate!(param_specs, optimizer, loss_config, training_config) → TrainingResult`

Lives in `training.jl`. The single main entry point.

```julia
result = calibrate!(
    param_specs,
    Optimisers.Adam(5f-3),
    TRENBERTH_LOSS,
    TrainingConfig(spinup_days=180, max_batches=300)
)
```

Internally:
1. Builds model, applies initial parameter values, initialises simulation
2. Sigmoid-reparameterises parameters into unconstrained space
3. Spinup loop
4. Batch loop:
   a. Advance `steps_per_sample` steps, call `compute_gradients!`, accumulate
   b. Average gradients; apply `grad_scale` and sigmoid chain-rule factor
   c. Gradient clip, `Optimisers.update!`, convert back to physical space
   d. `SpeedyWeather.reconstruct` model with new parameters
   e. Record history
   f. LR decay and early-stopping checks
5. Return `TrainingResult`

### `run_climate_validation(result; n_years, stat_years) → ClimateStats`

Lives in `validation.jl`. Builds a fresh climate simulation with the trained
parameter values, runs it with `DailyMeansCallback`, and returns equilibrium means.
Used for the post-training figures.

---

## Module File (`SpeedyCalibration.jl`)

```julia
module SpeedyCalibration

using SpeedyWeather, Enzyme, Optimisers
using Statistics, Printf, Dates, RingGrids

include("param_spec.jl")
include("loss.jl")
include("gradients.jl")
include("training.jl")
include("callbacks.jl")
include("validation.jl")
include("analysis.jl")

export ParamSpec
export LossConfig, OSR_LOSS, OSR_SRU_SRD_LOSS, TRENBERTH_LOSS
export TrainingConfig, quick_test_config
export TrainingResult
export enzyme_warmup, calibrate!
export DailyMeansCallback, run_climate_validation
export plot_training, plot_climate           # requires CairoMakie (weak dep)

end
```

CairoMakie is a **weak dependency** (declared in `[weakdeps]` and `[extensions]` in
`Project.toml`) so the package loads without it. Plotting functions are only available
after `using CairoMakie`.

---

## Project.toml (sketch)

```toml
name    = "SpeedyCalibration"
uuid    = "..."
version = "0.1.0"

[deps]
Enzyme     = "7da242da-08ed-463a-9acd-ee780be4f1d9"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
SpeedyWeather = "..."
Statistics = "..."
Printf     = "..."
Dates      = "..."
JLD2       = "..."

[weakdeps]
CairoMakie = "..."

[extensions]
SpeedyCalibrationMakieExt = "CairoMakie"

[compat]
julia = "1.10"
SpeedyWeather = "0.12"
```

---

## Design Decisions

### Why `LossConfig` instead of separate gradient functions?

The notebooks have `compute_gradients_osr_sru!` and `compute_gradients_trenberth!`
which are 90% identical. Every new flux combination would require another copy.
`LossConfig` holds the targets/weights/field-paths and `compute_gradients!` iterates
over them generically, so adding a new loss (e.g. including precipitation) is a
two-line `LossConfig` definition, not a new 80-line function.

### Why `TrainingConfig` struct instead of keyword args?

The training functions have ~20 kwargs. Passing them between functions, saving them
to disk alongside results, and documenting defaults becomes unmanageable. A struct
makes the configuration a first-class value: it can be serialised, diffed across
experiments, and passed around without forwarding every kwarg individually.

### Why sigmoid reparameterisation by default?

Bounds clamping after an optimizer step introduces discontinuities that confuse
Adam's momentum. The sigmoid maps `(-∞, +∞) → (lb, ub)` so the optimizer always
operates in unconstrained space. The chain-rule factor `∂θ_phys/∂θ_raw` shrinks
steps naturally as a parameter approaches its physical limit. This is the
approach already proven in the notebooks.

### Why `TrainingResult` instead of a bare tuple?

Returning `(history, final_params, conv_info)` forces callers to unpack positionally
and makes it impossible to add new output fields without breaking existing code.
A struct adds named access, backwards-compatible extension, and clean JLD2
serialisation for resume-safe long runs.

### What stays out of scope for v0.1

- GPU training (Enzyme AD is CPU-only today; GPU is only for validation)
- Neural network parameterisation replacement
- Multi-model ensemble calibration
- Automatic parameter discovery (the `find_all_trainable_parameters` exploration
  function from the notebooks is useful interactively but not a stable API)

---

## Migration Path from Notebooks

| Notebook concept            | Package equivalent                        |
|-----------------------------|-------------------------------------------|
| `ParamSpec` struct          | `SpeedyCalibration.ParamSpec` (identical) |
| `TRENBERTH_TARGETS/WEIGHTS` | `SpeedyCalibration.TRENBERTH_LOSS`        |
| `compute_gradients_*!`      | `SpeedyCalibration.compute_gradients!`    |
| `train_osr_sru_v6/v7`       | `SpeedyCalibration.calibrate!`            |
| `train_trenberth_v1/v3`     | `SpeedyCalibration.calibrate!`            |
| `DailyMeansCallback`        | `SpeedyCalibration.DailyMeansCallback`    |
| `analyze_training`          | `SpeedyCalibration.plot_training`         |
| `build_climate_sim` + loop  | `SpeedyCalibration.run_climate_validation`|

Notebooks become thin wrappers:

```julia
using SpeedyCalibration, Optimisers

enzyme_warmup()

result = calibrate!(
    training_params,              # same Vector{ParamSpec} as before
    Optimisers.Adam(5f-3),
    TRENBERTH_LOSS,
    TrainingConfig(spinup_days=180, max_batches=300)
)

figs = plot_training(result)
clm  = run_climate_validation(result; n_years=7, stat_years=5)
```

---

## Implementation Order

1. `param_spec.jl` — pure Julia, no deps, easy to test
2. `loss.jl` — `LossConfig`, predefined constants, flux extraction helpers
3. `gradients.jl` — `compute_gradients!` + `enzyme_warmup`
4. `callbacks.jl` — `DailyMeansCallback` (copy from notebook, no changes needed)
5. `training.jl` — `calibrate!` (consolidates v6/v7/v1/v3 into one function)
6. `validation.jl` — `run_climate_validation`
7. `analysis.jl` — `plot_training`, `plot_climate` (weak-dep extension)
8. `test/` — unit + smoke tests
9. `examples/` — reproduce two main experiments end-to-end