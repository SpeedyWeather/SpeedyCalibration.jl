# Group B2 of the 2026-08-01 plan: climate-validate the winning lrd weight=0.7 config
# found by the finer sweep (Group B1) -- fixed-yardstick score 96.0, best of 7 tested
# weights (0.3/0.4/0.5/0.6/0.7/0.8/1.0), a clean local optimum (0.6 and 0.8 both worse).
# See project_trenberth_lw_transmissivity_gradscale_fix memory, "CORRECTION 2026-08-01"
# section, for why this equilibrium check matters: training-metric wins have already been
# shown to not always survive contact with the true 7-year equilibrium (srd for the 0.3
# baseline looked perfect in training but was worse than default at equilibrium).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using SpeedyCalibration, Dates, Printf

r = load_result(joinpath(@__DIR__, "output", "trenberth_bd2_lrdweight07", "result.jld2"))
println("lrd weight=0.7  best_batch=", r.conv_info.best_batch,
        "  best_smoothed_loss=", round(r.conv_info.best_smoothed_loss, digits=2))

clm = run_climate_validation(r; n_years=7, stat_years=5, dt=Minute(20))

targets = r.loss_config.targets
@printf("\n%-6s  %8s  %9s  %9s  %9s  %9s\n", "flux", "target", "def val", "def bias", "trn val", "trn bias")
println("-" ^ 60)
for k in r.loss_config.flux_keys
    tgt   = targets[k]
    d_val = getproperty(clm.default, k)
    t_val = getproperty(clm.trained, k)
    @printf("%-6s  %8.2f  %9.2f  %+9.2f  %9.2f  %+9.2f\n", k, tgt, d_val, d_val-tgt, t_val, t_val-tgt)
end
@printf("\nPrecipitation:  default = %.2f mm/day  trained = %.2f mm/day  (ERA5 ≈ 2.74)\n",
        clm.default.precip_total, clm.trained.precip_total)
println("\nDone.")
