# Reproduces trenberth_full.ipynb's Sections 6-8 for the FINAL production config
# (batch_days=2, lrd weight=0.7) against the real output/trenberth_full_result.jld2
# (copied from the validated examples/trenberth_bd2_lrdweight07.jl run). Same
# reasoning as trenberth_full_downstream.jl: no jupyter/nbconvert available in this
# environment to execute the notebook directly, so this produces the real numbers
# and plots the notebook's cells would show once actually opened and run.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpeedyCalibration, CairoMakie, GeoMakie, Dates, Printf

save_path = joinpath(@__DIR__, "output", "trenberth_full_result.jld2")
result = load_result(save_path)

println("="^70)
println(result)
println()
println("Convergence info:")
println("  stop_reason:        ", result.conv_info.stop_reason)
println("  best_batch:         ", result.conv_info.best_batch)
println("  best_smoothed_loss: ", round(result.conv_info.best_smoothed_loss, digits=2))
println("  total_batches:      ", result.conv_info.total_batches)
@printf("  total_time:         %.2f hours\n", result.conv_info.total_time / 3600)

@printf("\n%-28s  %10s  %10s  %10s\n", "parameter", "initial", "best", "change")
println("-" ^ 65)
for spec in result.param_specs
    init    = isnothing(spec.initial) ? NaN32 : spec.initial
    trained = result.best_params[spec.name]
    @printf("%-28s  %10.4g  %10.4g  %+10.4g\n",
            spec.name, init, trained, trained - init)
end

@printf("\n%-6s  %8s  %8s  %8s\n", "flux", "target", "best", "bias")
println("-" ^ 38)
for k in result.loss_config.flux_keys
    tgt  = result.loss_config.targets[k]
    val  = result.history[k][result.conv_info.best_batch]
    @printf("%-6s  %8.2f  %8.2f  %+8.2f\n", k, tgt, val, val - tgt)
end

println("\nPlotting training history...")
figs = plot_training(result; save_dir=joinpath(@__DIR__, "output", "trenberth_full"))
println("  saved to output/trenberth_full/fig_{loss,flux,params,grads}.pdf")

println("\nRunning climate validation (default vs. best_params, n_years=7, stat_years=5)...")
clm = run_climate_validation(result; n_years=7, stat_years=5, dt=Minute(20))

targets = result.loss_config.targets
@printf("\n%-6s  %8s  %9s  %9s  %9s  %9s\n",
        "flux", "target", "def val", "def bias", "trn val", "trn bias")
println("-" ^ 60)
for k in result.loss_config.flux_keys
    tgt    = targets[k]
    d_val  = getproperty(clm.default, k)
    t_val  = getproperty(clm.trained, k)
    @printf("%-6s  %8.2f  %9.2f  %+9.2f  %9.2f  %+9.2f\n",
            k, tgt, d_val, d_val-tgt, t_val, t_val-tgt)
end
println()
println("Held-out diagnostics (not in loss):")
@printf("  Precipitation:  default = %.2f mm/day  trained = %.2f mm/day  (ERA5 ≈ 2.74)\n",
        clm.default.precip_total, clm.trained.precip_total)

println("\nPlotting climate validation...")
cfigs = plot_climate(clm;
    save_dir    = joinpath(@__DIR__, "output", "trenberth_full"),
    loss_config = result.loss_config,
)
println("  saved to output/trenberth_full/fig_clm_*.pdf")
println("\nDone.")
