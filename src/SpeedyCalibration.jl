"""
Gradient-based parameter calibration for SpeedyWeather.jl via online statistical
gradient estimation: running the model continuously and averaging single-timestep
Enzyme.jl gradients, instead of differentiating through long, chaotically unstable
trajectories.
"""
module SpeedyCalibration

using SpeedyWeather
using Enzyme
using Optimisers
using Statistics
using Printf
using Dates
using RingGrids
using JLD2
using NCDatasets
using Checkpointing

include("param_spec.jl")
include("loss.jl")
include("gradients.jl")
include("gradients_checkpointed.jl")
include("callbacks.jl")
include("diagnostics.jl")
include("reference_data.jl")
include("training.jl")
include("validation.jl")
include("analysis.jl")

export ParamSpec
export get_by_path, set_by_path!, to_raw, sigmoid_param, sigmoid_grad_factor
export LossConfig, OSR_LOSS, OSR_SRU_SRD_LOSS, TRENBERTH_LOSS, make_normalized_loss
export TrainingConfig, quick_test_config
export TrainingResult, save_result, load_result, save_artifacts
export enzyme_warmup, calibrate!
export compute_gradients_checkpointed!
export DailyMeansCallback
export build_climate_sim, run_climate_validation
export plot_training, plot_climate, plot_experiment_comparison
export check_nans, vertical_gradient_anomalies, NaNWatchCallback, any_anomaly
export plot_nan_watch, plot_vertical_profile
export load_netcdf_field, time_mean, regrid_nearest, compare_to_reference
export plot_field_map, plot_native_field, plot_comparison_map

end # module
