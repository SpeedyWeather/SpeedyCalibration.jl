# Quick start

This page walks you through a minimal end-to-end calibration — from defining what
to tune to reading the trained parameter values. A smoke-test-sized run (trunc=5,
5 batches, 5-day spinup) completes in a few minutes and is a good first check that
everything works on your machine.

## Enzyme warmup

Enzyme.jl compiles its AD rules the first time it encounters a function signature.
This compilation takes a few minutes and would otherwise happen silently inside the
first training batch, making it appear to hang.

`calibrate!` handles this automatically. When `TrainingConfig.warmup_enzyme = true`
(the default), it runs one backward pass on the **actual training model** right after
initialisation, before the spinup. You will see printed output like:

```
Warming up Enzyme (compiling AD rules on actual model)...
Enzyme warmup complete in 47.3 s.
```

The warmup must use the same model that training will use — a warmup on a different
resolution or type is useless because Enzyme's compiled rules are specialised to the
exact Julia types of `variables` and `model`. If Enzyme is already compiled from an
earlier run in the same Julia session, skip it:

```julia
TrainingConfig(warmup_enzyme=false, ...)
```

## Define trainable parameters

A [`ParamSpec`](@ref) describes one trainable parameter: where it lives in the model
tree, its physical bounds, and optionally an initial value that overrides the model
default.

```@example quickstart
params = [
    ParamSpec(:cloud_albedo,
              [:shortwave_radiation, :clouds, :cloud_albedo];
              bounds  = (0.1f0, 0.85f0),
              initial = 0.35f0),
]
```

The `path` argument mirrors Julia's property access syntax: the path
`[:shortwave_radiation, :clouds, :cloud_albedo]` corresponds to
`model.shortwave_radiation.clouds.cloud_albedo`.
See [Defining parameters](@ref) for details on how to discover paths, set
`grad_scale` for numerically weak gradients, and add multiple parameters.

## Choose a loss

SpeedyCalibration ships three ready-made loss configurations based on the
[Trenberth et al. (2009)](https://doi.org/10.1175/2008BAMS2634.1) global energy
budget:

| Constant | Fluxes | Use case |
|---|---|---|
| `OSR_LOSS` | outgoing shortwave | quick single-target calibration |
| `OSR_SRU_SRD_LOSS` | outgoing + surface SW | SW budget closure |
| `TRENBERTH_LOSS` | all 6 SW+LW fluxes | full energy-balance calibration |

For the smoke test we use `OSR_LOSS`:

```@example quickstart
loss_config = OSR_LOSS
```

See [Loss configuration](@ref) for how to define a custom loss with your own targets
and weights.

## Run calibration

[`calibrate!`](@ref) is the main entry point. Pass the parameter specs, an
Optimisers.jl optimizer, the loss config, and a [`TrainingConfig`](@ref):

```@example quickstart
result = calibrate!(
    params,
    Optimisers.Adam(1f-2),
    loss_config,
    quick_test_config(),   # trunc=5, spinup=5d, max_batches=5
)
```

`quick_test_config()` is a convenience function that returns a minimal
[`TrainingConfig`](@ref) for fast iteration. For a full run, build a
`TrainingConfig` explicitly — see [Training](@ref).

While training, SpeedyCalibration prints a progress table:

```
Batch   1 | LR 1.0e-02 | osr= 93.4 | L̄   574.89 | Δp 1.2e-02
Batch   2 | LR 1.0e-02 | osr= 94.1 | L̄   528.44 | Δp 9.3e-03
...
```

## Inspect results

`calibrate!` returns a [`TrainingResult`](@ref):

```@example quickstart
result.final_params        # Dict{Symbol,Float32} of trained values
result.conv_info           # convergence summary NamedTuple
result.history[:osr]       # OSR time series, one entry per batch
result.history[:cloud_albedo]  # parameter trajectory
```

The full history dictionary contains the loss curve, smoothed loss, all flux
time series, parameter trajectories, and per-parameter gradient means and standard
deviations. See [Training](@ref) for a complete description of what is recorded.

## Save and load

Training results can be saved to JLD2 for later analysis:

```julia
save_result(result, "my_run.jld2")
result2 = load_result("my_run.jld2")
```

## Plot (optional)

With CairoMakie loaded, `plot_training` produces four figures — loss curve, flux
trajectories, parameter trajectories, and gradient magnitudes:

```julia
using CairoMakie

figs = plot_training(result; save_dir="output/")
figs.fig_loss    # loss + LR schedule
figs.fig_flux    # per-flux time series vs targets
figs.fig_params  # parameter trajectories with bounds
figs.fig_grads   # gradient mean ± std per parameter
```

## Next steps

- [Defining parameters](@ref) — how to find parameter paths, set `grad_scale`,
  choose initial values
- [Loss configuration](@ref) — custom targets, Trenberth constants, multi-flux losses
- [Training](@ref) — `TrainingConfig` fields explained, LR decay, early stopping
- [Validation](@ref) — run a post-training climate simulation and compare to defaults
