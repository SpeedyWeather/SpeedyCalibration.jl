# trenberth_full_cloudabs.jl completed training successfully (400 batches, best_smoothed_loss=52.82)
# and saved the result, but crashed on plot_training(::TrainingResult; save_dir::String) --
# "no method matching". Root cause: plot_training's actual method lives in
# ext/SpeedyCalibrationMakieExt.jl, a package extension that only activates when BOTH CairoMakie
# AND GeoMakie are loaded (Project.toml [weakdeps] / [extensions]: SpeedyCalibrationMakieExt =
# ["CairoMakie", "GeoMakie"]) -- the training script only had `using CairoMakie`, missing GeoMakie
# (trenberth_full.ipynb's own setup cell has both; this one didn't). No retraining needed --
# load the already-saved result and run the remaining steps (climate validation + plots) here.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using SpeedyCalibration, CairoMakie, GeoMakie, Dates, Printf

result = load_result(joinpath(@__DIR__, "output", "trenberth_full_cloudabs_result.jld2"))
println(result)

figs = plot_training(result; save_dir=joinpath(@__DIR__, "output"))
println("Training plots saved.")

clm = run_climate_validation(result; n_years=7, stat_years=5, dt=Minute(20))
targets = result.loss_config.targets
@printf("\n%-6s  %8s  %9s  %9s  %9s  %9s\n", "flux", "target", "def val", "def bias", "trn val", "trn bias")
println("-" ^ 60)
for k in result.loss_config.flux_keys
    tgt    = targets[k]
    d_val  = getproperty(clm.default, k)
    t_val  = getproperty(clm.trained, k)
    @printf("%-6s  %8.2f  %9.2f  %+9.2f  %9.2f  %+9.2f\n", k, tgt, d_val, d_val-tgt, t_val, t_val-tgt)
end
println()
println("Held-out diagnostics (not in loss):")
@printf("  Precipitation:  default = %.2f mm/day  trained = %.2f mm/day  (ERA5 ≈ 2.74)\n",
        clm.default.precip_total, clm.trained.precip_total)

cfigs = plot_climate(clm; save_dir=joinpath(@__DIR__, "output"), loss_config=result.loss_config)
println("Climate plots saved.")
