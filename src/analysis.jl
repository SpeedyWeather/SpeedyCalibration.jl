# Stub — plotting lives in ext/SpeedyCalibrationMakieExt.jl (weak dependency).
# These functions are defined only when CairoMakie is loaded.

"""
    plot_training(result; kwargs...) → Figure

Plot training history: loss curve, parameter trajectories, gradient magnitudes.
Requires `using CairoMakie`.
"""
function plot_training end

"""
    plot_climate(validation; kwargs...) → NamedTuple{:rad, :lw, :precip, :temp, :summary}

Plot climate validation: SW budget, LW budget, precipitation, temperature profile,
Trenberth summary. Requires `using CairoMakie`.
"""
function plot_climate end

"""
    plot_experiment_comparison(results::Vector{TrainingResult}; kwargs...) → NamedTuple

Compare multiple training runs: overlaid loss curves and final parameter bar chart.
Requires `using CairoMakie`.
"""
function plot_experiment_comparison end

"""
    plot_vertical_profile(profile; kwargs...) → Figure

Plot a vertical profile (e.g. temperature) with anomalous-gradient layers flagged.
Requires `using CairoMakie`.
"""
function plot_vertical_profile end

"""
    plot_nan_watch(cb::NaNWatchCallback; kwargs...) → Figure

Plot non-finite/blow-up counts over time for fields tracked by a `NaNWatchCallback`.
Requires `using CairoMakie`.
"""
function plot_nan_watch end

"""
    plot_field_map(lons, lats, data; kwargs...) → Figure

Plot a single lon/lat-gridded field as a global heatmap. Requires `using CairoMakie`.
"""
function plot_field_map end

"""
    plot_native_field(field; kwargs...) → Figure

Plot a RingGrids field on its native grid as cell polygons. Requires `using CairoMakie`.
"""
function plot_native_field end

"""
    plot_comparison_map(sim, ref, lons, lats; kwargs...) → Figure

Three-panel sim/reference/difference comparison map. Requires `using CairoMakie`.
"""
function plot_comparison_map end
