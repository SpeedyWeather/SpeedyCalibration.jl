# Validation

After calibration you want to know whether the trained parameters actually improve
the model's climatology, not just the loss during training. This page covers
climate validation runs and the `DailyMeansCallback` that powers them.

## Why post-training validation?

The training loop optimises a loss computed from *short* (3-day) windows. A parameter
set that drives the OSR toward its target on short timescales may or may not produce
the right equilibrium climatology over years. It is therefore important to run a
separate, longer simulation with the trained parameters and compare the equilibrium
statistics against both the default model and the Trenberth targets.

## Running a climate validation

[`run_climate_validation`](@ref) builds two fresh simulations, one with default
parameters and one with the trained values, runs each for `n_years` years, and
returns equilibrium statistics over the last `stat_years`:

```julia
clm = run_climate_validation(result; n_years=7, stat_years=5)
```

The return value has the structure:

```julia
clm.default       # equilibrium stats for the default model
clm.trained       # equilibrium stats for the trained model
clm.default_cb    # DailyMeansCallback from the default run (full time series)
clm.trained_cb    # DailyMeansCallback from the trained run (full time series)
```

Each stats namedtuple contains:

| Field | Unit | Description |
|---|---|---|
| `osr` | W m⁻² | Equilibrium mean outgoing shortwave |
| `olr` | W m⁻² | Equilibrium mean outgoing longwave |
| `srd` | W m⁻² | Equilibrium mean surface SW down |
| `sru` | W m⁻² | Equilibrium mean surface SW up |
| `lrd` | W m⁻² | Equilibrium mean surface LW down |
| `lru` | W m⁻² | Equilibrium mean surface LW up |
| `precip_total` | mm day⁻¹ | Equilibrium mean global precipitation |
| `temp_profile` | K | Equilibrium mean temperature, one value per model level |

Print a quick bias table:

```julia
targets = TRENBERTH_LOSS.targets

println("Flux          Default bias   Trained bias")
for k in [:osr, :olr, :srd, :sru, :lrd, :lru]
    t = targets[k]
    d_bias = getproperty(clm.default, k) - t
    t_bias = getproperty(clm.trained, k) - t
    @printf("%-12s   %+8.2f       %+8.2f\n", k, d_bias, t_bias)
end
```

## Plotting validation results

With CairoMakie loaded, `plot_climate` produces five figures:

```julia
using CairoMakie

cfigs = plot_climate(clm; save_dir="output/")
cfigs.fig_rad      # SW budget time series (OSR, SRD, SRU)
cfigs.fig_lw       # LW budget time series (OLR, LRD, LRU)
cfigs.fig_precip   # precipitation time series
cfigs.fig_summary  # side-by-side bias bar chart
```

Each figure shows default (grey) and trained (blue) together with the Trenberth
reference value (dashed black). The bias bar chart is the clearest summary: bars
above zero mean the model over-shoots the target, below zero means under-shoot.
A successful calibration should move the trained bars closer to zero for the
fluxes that were in the loss.

!!! note "Held-out diagnostics"
    Always check fluxes that were *not* in the loss as well. It is common for
    optimising a subset of fluxes to shift biases elsewhere. For example, lowering
    cloud albedo to reduce OSR may increase surface SW down (SRD) beyond its target.
    Post-training validation is the only way to detect these compensatory effects.

## `DailyMeansCallback`

The climate runs use [`DailyMeansCallback`](@ref) internally, but you can also
attach it to any SpeedyWeather simulation directly for custom monitoring:

```julia
using SpeedyWeather, SpeedyCalibration

sg    = SpectralGrid(trunc=31, nlayers=8)
model = PrimitiveWetModel(sg)
cb    = DailyMeansCallback(sg)
add!(model, :daily_means => cb)

sim = initialize!(model)
run!(sim, Day(365))

cb.osr    # Vector{Float32} of daily OSR means
cb.days   # Vector{Float64} of simulation days
```

The callback records one value per day (via `Schedule(every=Day(1))`) for each of:

| Field | Description |
|---|---|
| `days` | Day number since simulation start |
| `temp` | Temperature profile (time × layer) in K |
| `olr` | Outgoing longwave radiation [W m⁻²] |
| `osr` | Outgoing shortwave radiation [W m⁻²] |
| `srd` | Surface SW down [W m⁻²] |
| `sru` | Surface SW up [W m⁻²] |
| `lrd` | Surface LW down [W m⁻²] |
| `lru` | Surface LW up [W m⁻²] |
| `precip_total` | Total precipitation [mm day⁻¹] |
| `precip_conv` | Convective precipitation [mm day⁻¹] |
| `precip_ls` | Large-scale precipitation [mm day⁻¹] |
| `cloud_top` | Global mean cloud-top level index |

All values are cosine-latitude weighted global means, computed identically to
the loss function, so comparing `cb.osr` to `TRENBERTH_LOSS.targets[:osr]` is
a direct measurement of how well the training has worked.

## Building a custom validation simulation

[`build_climate_sim`](@ref) gives you a bare `Simulation` with your trained parameters
applied, which you can instrument however you like:

```julia
sim = build_climate_sim(result.final_params, result.param_specs;
                        trunc=31, nlayers=8)

# attach your own callbacks
add!(sim.model, :my_cb => MyCustomCallback())

run!(sim, Day(7*365); output=false)
```

This is useful if you want to record diagnostics that `DailyMeansCallback` does not
cover, or if you want to run at a different resolution than the training run.
