```@raw html
---
layout: home

hero:
  name: "SpeedyCalibration.jl"
  tagline: "Gradient-based parameter calibration for SpeedyWeather.jl."
  actions:
    - theme: brand
      text: Quick start
      link: /quickstart
    - theme: alt
      text: View on GitHub
      link: https://github.com/SpeedyWeather/SpeedyCalibration.jl
    - theme: alt
      text: API Reference
      link: /api
---
```

SpeedyCalibration.jl implements **online statistical gradient estimation** for parameter
calibration in [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl).
Instead of differentiating through a long, chaotically unstable simulation trajectory,
it runs the model *continuously* and averages single-timestep
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) gradients to get a stable,
memory-efficient estimate of ∂L/∂θ.

## What problem does this solve?

SpeedyWeather's parameterisations contain dozens of free parameters (cloud albedos,
longwave emissivities, convective time scales) that are currently set by hand. Hand-tuning
is opaque, hard to reproduce, and does not generalise across model configurations.
Automatic differentiation can compute gradients of any scalar loss (e.g. bias in
top-of-atmosphere shortwave flux) with respect to these parameters, enabling gradient
descent optimisation. The main technical obstacle is that chaotic dynamics cause
gradients of *long* trajectories to explode exponentially. SpeedyCalibration sidesteps
this by never differentiating through more than one timestep.

## Overview

A typical calibration run looks like this:

```julia
using SpeedyCalibration, Optimisers

enzyme_warmup()   # compile Enzyme AD rules once per session

params = [
    ParamSpec(:cloud_albedo, [:shortwave_radiation, :clouds, :cloud_albedo];
              bounds=(0.1f0, 0.85f0)),
    ParamSpec(:ozone_absorption, [:shortwave_radiation, :ozone_absorption];
              bounds=(0.001f0, 0.1f0)),
]

result = calibrate!(params, Optimisers.Adam(5f-3), TRENBERTH_LOSS,
                    TrainingConfig(spinup_days=180, max_batches=300))
```

SpeedyCalibration takes care of the Enzyme AD pass, state save/restore, sigmoid
reparameterisation to enforce physical bounds, gradient clipping, learning rate decay,
and history tracking. You define which parameters to tune, what the targets are, and how
long to run; everything else is handled automatically.

## Key concepts

| Concept | Type | Description |
|---|---|---|
| Trainable parameter | `ParamSpec` | Name, path into model, physical bounds, initial value |
| Loss function | `LossConfig` | Which fluxes to target, Trenberth reference values, weights |
| Hyperparameters | `TrainingConfig` | Spinup, batch size, learning rate schedule, stopping criteria |
| Results | `TrainingResult` | Full history, final parameters, convergence info; serialisable to JLD2 |

## Method summary

At each training batch, SpeedyCalibration:

1. Advances the running simulation by `steps_per_sample` timesteps
2. Computes the area-weighted global mean of each tracked flux
3. Seeds the adjoint with `∂L/∂flux` cotangents, distributed per grid ring
4. Calls `Enzyme.autodiff(Reverse, timestep!, ...)` for a single backward pass
5. Saves the resulting `∂L/∂θ`, then *restores* the full prognostic state so the
   forward simulation continues from exactly where it was
6. After `samples_per_batch` samples, averages the gradients and takes an Adam step
   in unconstrained (sigmoid-reparameterised) space

The averaging over many samples within a batch is what stabilises the gradient
estimate under chaotic dynamics: individual timestep gradients are noisy, but their
mean converges to the correct sensitivity.

## Citing

If you use SpeedyCalibration.jl in research or teaching, please cite the associated
master thesis:

> Viebig, N. (2026). *Gradient-based parameter calibration for SpeedyWeather.jl via
> online statistical gradient estimation.* MSc thesis, ETH Zurich.
> [doi:10.3929/ethz-c-000799367](https://doi.org/10.3929/ethz-c-000799367)

and the underlying SpeedyWeather paper:

> Klöwer et al. (2024). SpeedyWeather.jl: Reinventing atmospheric general circulation
> models towards interactivity and extensibility. *Journal of Open Source Software*,
> **9(98)**, 6323. doi:10.21105/joss.06323
