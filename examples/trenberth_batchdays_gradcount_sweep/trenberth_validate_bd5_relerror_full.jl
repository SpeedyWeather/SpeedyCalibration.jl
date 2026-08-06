# True 7-year equilibrium validation for the batch_days=5 + relative-error
# weighting result (examples/trenberth_batchdays_sweep/trenberth_bd5_relerror.jl).
# First time bd=5 has been trained to real convergence (patience-based stop,
# not a fixed truncation) under this weighting. See
# trenberth_bd5_relerror.jl's header comment for why bd=10 was ruled out
# first: its own full history shows lrd looking great early (~240 simulated
# days) then drifting monotonically worse for the rest of training, ending
# up worse than default. bd=5 is the genuinely untested case.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using SpeedyCalibration, Dates, Printf

r = load_result(joinpath(@__DIR__, "..", "output", "trenberth_bd5_relerror", "result.jld2"))
println("batch_days=5, relative-error weight  best_batch=", r.conv_info.best_batch,
        "  best_smoothed_loss=", round(r.conv_info.best_smoothed_loss, digits=2))

clm = run_climate_validation(r; n_years=7, stat_years=5, dt=Minute(20))

println("Climate run: default ...")
println("Climate run: trained ...")
println()
println("flux      target    def val   def bias    trn val   trn bias")
println("-" ^ 62)
for k in r.loss_config.flux_keys
    tgt = r.loss_config.targets[k]
    dv  = getproperty(clm.default, k)
    tv  = getproperty(clm.trained, k)
    @printf("%-6s  %8.2f  %8.2f  %+9.2f  %8.2f  %+9.2f\n", k, tgt, dv, dv - tgt, tv, tv - tgt)
end

println()
println("Precipitation:  default = ", round(clm.default.precip_total, digits=2),
        " mm/day  trained = ", round(clm.trained.precip_total, digits=2), " mm/day  (ERA5 ≈ 2.74)")
println()
println("Done.")
