# SpeedyCalibration

Gradient-based parameter calibration for [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) — systematic, reproducible, and fast.

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://speedyweather.github.io/SpeedyCalibration.jl/dev)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue.svg)](LICENSE)

SpeedyCalibration.jl implements **online statistical gradient estimation** for tuning
SpeedyWeather's free parameterisation parameters (cloud albedos, longwave emissivities,
convective time scales, ...) against observational targets, instead of hand-tuning them.
Rather than differentiating through a long — and chaotically unstable — simulation
trajectory, it runs the model continuously and averages single-timestep
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) gradients to get a stable,
memory-efficient estimate of ∂L/∂θ.

## Installation

Not yet registered in the General registry — install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/SpeedyWeather/SpeedyCalibration.jl")
```

## Quick start

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

SpeedyCalibration handles the Enzyme AD pass, state save/restore, sigmoid
reparameterisation to enforce physical bounds, gradient clipping, learning rate decay,
and history tracking. You define which parameters to tune, what the targets are, and
how long to run.

See the [documentation](https://speedyweather.github.io/SpeedyCalibration.jl/dev) for the
full guide, API reference, and diagnostic/plotting tools (anomaly detection, vertical
profile checks, satellite-reference comparison).

## Citing

If you use SpeedyCalibration.jl in research or teaching, please cite the associated
master thesis:

> Viebig, N. (2026). *Gradient-based parameter calibration for SpeedyWeather.jl via
> online statistical gradient estimation.* MSc thesis, [University].

and the underlying SpeedyWeather paper:

> Klöwer et al. (2024). SpeedyWeather.jl: Reinventing atmospheric general circulation
> models towards interactivity and extensibility. *Journal of Open Source Software*,
> **9(98)**, 6323. doi:10.21105/joss.06323

## License

EUPL-1.2, matching [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl). See [LICENSE](LICENSE).
