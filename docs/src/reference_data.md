# Reference-data utilities

SpeedyCalibration provides small utilities for validation against NetCDF-based reference products.

## Load fields from NetCDF

[`load_netcdf_field`](@ref) reads a variable and its lon/lat coordinates, converts `missing` to `NaN`, and supports 1D/2D/3D data:

```julia
data, lons, lats = load_netcdf_field("obs.nc", "osr")
```

## Time averaging

[`time_mean`](@ref) reduces `(lon, lat, time)` arrays to `(lon, lat)` means:

```julia
monthly_mean = time_mean(data)
```

Optional weights can be passed for weighted averages.

## Regridding

[`regrid_nearest`](@ref) maps fields between grids via nearest-neighbor lookup, including longitude wrap-around handling:

```julia
ref_on_model_grid = regrid_nearest(ref_data, ref_lons, ref_lats, model_lons, model_lats)
```

## Quantitative comparison

[`compare_to_reference`](@ref) computes pointwise comparison statistics:

```julia
stats = compare_to_reference(sim_field, ref_on_model_grid)
stats.rmse
stats.mean_bias
```

Returned metrics include sample count, means, bias, RMSE, max absolute error, and correlation.
