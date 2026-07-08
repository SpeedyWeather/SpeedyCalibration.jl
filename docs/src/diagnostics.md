# Diagnostics

SpeedyCalibration includes lightweight diagnostics for numerical stability checks during development and long runs.

## NaN/Inf checks

Use [`check_nans`](@ref) for a one-shot summary of any array:

```julia
stats = check_nans(sim.variables.grid.temperature; label="temperature")
stats.ok
stats.frac_bad
```

Returned fields include total count, NaN/Inf counts, finite min/max, and an `ok` flag.

## Vertical profile anomaly detection

[`vertical_gradient_anomalies`](@ref) highlights abrupt layer-to-layer jumps in vertical profiles:

```julia
flags = vertical_gradient_anomalies(clm.trained.temp_profile; ratio_threshold=1.5)
```

This is useful for quickly spotting unstable temperature structures after training.

## Online anomaly monitoring callback

[`NaNWatchCallback`](@ref) tracks non-finite and blow-up counts over time for selected grid fields:

```julia
using SpeedyWeather, SpeedyCalibration

sg = SpectralGrid(trunc=31, nlayers=8)
model = PrimitiveWetModel(sg)
cb = NaNWatchCallback(fields=[:temperature, :humidity], bound=1e3)
add!(model, :nan_watch => cb)

sim = initialize!(model)
run!(sim, Day(30))

any_anomaly(cb)
```

If CairoMakie is loaded, [`plot_nan_watch`](@ref) visualizes anomaly counts and max magnitudes over time.
