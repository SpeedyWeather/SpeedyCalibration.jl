# API Reference

## Parameter specification

```@docs
ParamSpec
```

## Loss configuration

```@docs
LossConfig
OSR_LOSS
OSR_SRU_SRD_LOSS
TRENBERTH_LOSS
make_normalized_loss
```

## Training

```@docs
TrainingConfig
quick_test_config
TrainingResult
calibrate!
save_result
load_result
save_artifacts
```

## Gradients

```@docs
enzyme_warmup
compute_gradients!
```

## Callbacks

```@docs
DailyMeansCallback
```

## Diagnostics

```@docs
check_nans
vertical_gradient_anomalies
NaNWatchCallback
any_anomaly
```

## Reference data

```@docs
load_netcdf_field
time_mean
regrid_nearest
compare_to_reference
```

## Validation

```@docs
build_climate_sim
run_climate_validation
```

## Plotting

These functions are only available when `CairoMakie` is loaded.

```@docs
plot_training
plot_climate
plot_experiment_comparison
plot_nan_watch
plot_vertical_profile
plot_field_map
plot_native_field
plot_comparison_map
```
