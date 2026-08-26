# Full 6-flux Trenberth calibration, TRAINED under a real seasonal cycle -- not perpetual equinox.
#
# Context: `calibrate!` has always trained with seasonal_cycle hardcoded false (permanent equinox),
# while `run_climate_validation` defaults to seasonal_cycle=true (matching the thesis's own
# methodology) -- meaning every prior Trenberth run in this investigation was TRAINED against one
# climate state and VALIDATED against a different one. `examples/README.md`'s "Why not seasonal
# cycle next" section deliberately deferred this until the lrd/olr structural coupling problem was
# resolved, reasoning that seasonal cycle adds another slow timescale on top of an already
# poorly-understood slow-timescale problem, more likely to make lrd/srd drift worse than to
# diagnose anything. Per explicit user direction (2026-08-26), proceeding anyway now that
# trenberth_full_cloudabs.jl has produced a genuinely improved starting point (srd equilibrium
# bias down to +3.4, best of the whole investigation) -- warm-starting from ITS best_params rather
# than raw model defaults, both because it's a better starting climate and because retraining from
# scratch under a harder (seasonal) training signal is more likely to succeed from a good point.
#
# `seasonal_cycle=true` added to TrainingConfig in src/training.jl this session (was previously
# hardcoded false, not configurable at all).
#
# Design choices made explicit here (none of this has been tried before in this codebase):
# - spinup_days=730 (2 years, not the usual 180) -- 180 days from a March 21 start barely reaches
#   the opposite equinox, nowhere near enough for the land/snow/sea-ice seasonal cycle to
#   phase-lock into a repeating annual pattern before training starts. 2 years is a judgment call:
#   long enough to clear the initial-condition transient with this model's simplified mixed-layer
#   ocean (fast response vs. a full ocean GCM), short enough to keep spinup cost from dominating
#   the run. Not validated against a longer spinup -- worth revisiting if training looks like it's
#   chasing a still-drifting seasonal state.
# - batch_days=2/samples_per_batch=10 kept UNCHANGED from the validated production config. Unlike
#   the diurnal-aliasing risk (samples_per_batch tiling one day, needs gcd(steps_per_sample,
#   steps_per_day)=1), there is no equivalent seasonal-aliasing risk here: the simulation clock
#   advances continuously across batches (never resets), so different batches naturally land at
#   different points in the year as training progresses -- 400 batches * 2 days = 800 days ≈ 2.2
#   years of continuous forward motion, not a periodic re-sampling of the same season.
# - grad_scale values kept identical to trenberth_full_cloudabs.jl -- untested under seasonal
#   cycle specifically, but no a priori reason the relative raw-gradient-magnitude imbalance
#   between the LW-transmissivity block and the rest would change qualitatively just from adding
#   seasonal forcing. Flagged in case this run's own grad_norm/clipped diagnostics say otherwise.
#
# Expected runtime: longer than trenberth_full_cloudabs.jl's ~23 min, both from the 4x longer
# spinup (730 vs 180 days) and because the underlying physics (seasonal insolation swings) may
# make single-timestep gradients noisier -- no prior data point for how much longer, unlike the
# well-characterized permanent-equinox runs in this investigation.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using SpeedyCalibration, Optimisers, CairoMakie, GeoMakie, Dates, Printf

# Warm-started from trenberth_full_cloudabs's best_params (batch 391, best_smoothed_loss=52.82),
# not the raw model defaults -- see header. grad_scale values unchanged from that run.
param_specs = [
    ParamSpec(:cloud_albedo,
        [:shortwave_radiation, :clouds, :cloud_albedo];
        bounds=(0.25f0, 0.95f0), initial=0.8252078f0),
    ParamSpec(:stratocumulus_cover_max,
        [:shortwave_radiation, :clouds, :stratocumulus_cover_max];
        bounds=(0.25f0, 0.95f0), initial=0.8396477f0),
    ParamSpec(:stratocumulus_albedo,
        [:shortwave_radiation, :clouds, :stratocumulus_albedo];
        bounds=(0.10f0, 0.90f0), initial=0.7771105f0),
    ParamSpec(:precipitation_weight,
        [:shortwave_radiation, :clouds, :precipitation_weight];
        bounds=(0.0f0, 0.8f0), initial=0.55010515f0),
    ParamSpec(:absorptivity_water_vapor,
        [:shortwave_radiation, :transmissivity, :absorptivity_water_vapor];
        bounds=(60f0, 140f0), initial=64.29744f0, grad_scale=0.01f0),
    ParamSpec(:absorptivity_dry_air,
        [:shortwave_radiation, :transmissivity, :absorptivity_dry_air];
        bounds=(0.005f0, 0.060f0), initial=0.011957027f0),
    ParamSpec(:absorptivity_aerosol,
        [:shortwave_radiation, :transmissivity, :absorptivity_aerosol];
        bounds=(0.005f0, 0.060f0), initial=0.013911673f0),
    ParamSpec(:ozone_absorption,
        [:shortwave_radiation, :radiative_transfer, :ozone_absorption];
        bounds=(0.002f0, 0.020f0), initial=0.003880587f0),
    ParamSpec(:absorptivity_cloud_base,
        [:shortwave_radiation, :transmissivity, :absorptivity_cloud_base];
        bounds=(2f0, 30f0), initial=4.473114f0),
    ParamSpec(:albedo_land,
        [:albedo, :land, :albedo_land];
        bounds=(0.10f0, 0.70f0), initial=0.21267866f0),
    ParamSpec(:albedo_high_vegetation,
        [:albedo, :land, :albedo_high_vegetation];
        bounds=(0.04f0, 0.26f0), initial=0.11076762f0),
    ParamSpec(:albedo_low_vegetation,
        [:albedo, :land, :albedo_low_vegetation];
        bounds=(0.05f0, 0.35f0), initial=0.13833371f0),
    ParamSpec(:albedo_snow,
        [:albedo, :land, :albedo_snow];
        bounds=(0.15f0, 0.75f0), initial=0.6501491f0),
    ParamSpec(:snow_depth_scale,
        [:albedo, :land, :snow_depth_scale];
        bounds=(0.005f0, 0.20f0), initial=0.025206782f0),
    ParamSpec(:albedo_ocean,
        [:albedo, :ocean, :albedo_ocean];
        bounds=(0.02f0, 0.10f0), initial=0.09514533f0),
    ParamSpec(:albedo_ice,
        [:albedo, :ocean, :albedo_ice];
        bounds=(0.30f0, 0.90f0), initial=0.86301625f0),
    ParamSpec(:tau0_equator,
        [:longwave_radiation, :transmissivity, :τ₀_equator];
        bounds=(2f0, 12f0), initial=6.522244f0, grad_scale=0.04f0),
    ParamSpec(:tau0_pole,
        [:longwave_radiation, :transmissivity, :τ₀_pole];
        bounds=(0.3f0, 4f0), initial=0.5172882f0, grad_scale=0.25f0),
    ParamSpec(:fl,
        [:longwave_radiation, :transmissivity, :fₗ];
        bounds=(0.0f0, 0.5f0), initial=0.41680473f0, grad_scale=0.025f0),
    ParamSpec(:emissivity_ocean,
        [:longwave_radiation, :radiative_transfer, :emissivity_ocean];
        bounds=(0.80f0, 1.00f0), initial=0.92778087f0, grad_scale=0.2f0),
    ParamSpec(:emissivity_land,
        [:longwave_radiation, :radiative_transfer, :emissivity_land];
        bounds=(0.80f0, 1.00f0), initial=0.88495386f0, grad_scale=0.5f0),
]

println("$(length(param_specs)) trainable parameters defined (warm-started from trenberth_full_cloudabs).")

loss_config = LossConfig(
    [:osr, :sru, :srd, :olr, :lrd, :lru];
    targets = Dict(:osr => 101.9f0, :sru =>  23.0f0, :srd => 168.0f0,
                   :olr => 235.0f0, :lrd => 333.0f0, :lru => 398.0f0),
    weights = Dict(:osr => 1.00000f0, :sru => 19.62875f0, :srd => 0.36790f0,
                   :olr => 0.18802f0, :lrd => 0.09364f0,  :lru => 0.06555f0),
)

# Quick smoke test -- seasonal_cycle=true is brand new to this codebase's training path, verify
# it builds/runs end to end before committing to the full run. Note: quick_test_config()'s default
# spinup_days=5 is nowhere near enough to phase-lock a seasonal cycle -- fine for a wiring check,
# NOT a preview of real seasonal training dynamics.
result_test = calibrate!(
    param_specs, Optimisers.Adam(5f-3), loss_config,
    quick_test_config(; seasonal_cycle=true),
)
println("Quick test complete: ", result_test.conv_info.stop_reason)
println("\nGradient magnitudes (last batch):")
@printf("  %-28s  %12s\n", "parameter", "|mean grad|")
println("  " * "-" ^ 45)
for spec in result_test.param_specs
    g = result_test.history[Symbol("grad_", spec.name)]
    isempty(g) && continue
    @printf("  %-28s  %12.3e\n", spec.name, abs(g[end]))
end

# Full production run -- seasonal_cycle=true, spinup lengthened to 730 days (2 years) for phase-
# lock (see header). Everything else identical to trenberth_full_cloudabs.jl's config.
save_path = joinpath(@__DIR__, "output", "trenberth_seasonal_full_result.jld2")

result = calibrate!(
    param_specs,
    Optimisers.Adam(5f-3),
    loss_config,
    TrainingConfig(
        spinup_days       = 730,
        batch_days        = 2.0,
        samples_per_batch = 10,
        max_batches       = 400,
        loss_threshold    = 1f-6,
        enable_lr_decay   = false,
        trunc             = 31,
        nlayers           = 8,
        daily_cycle       = true,
        seasonal_cycle    = true,
    ),
)
mkpath(dirname(save_path))
save_result(result, save_path)
println("Result saved to: $save_path")

println(result)
println()
println("Convergence info:")
println("  stop_reason:        ", result.conv_info.stop_reason)
println("  best_smoothed_loss: ", round(result.conv_info.best_smoothed_loss, digits=2))
println("  total_batches:      ", result.conv_info.total_batches)
@printf("  total_time:         %.1f hours\n", result.conv_info.total_time / 3600)

if result.conv_info.total_batches == 0
    println("\nNo batches completed (stop_reason: $(result.conv_info.stop_reason)) -- ",
            "nothing to report below (history arrays are empty). Exiting.")
    exit(1)
end

@printf("\n%-28s  %10s  %10s  %10s\n", "parameter", "initial", "trained", "change")
println("-" ^ 65)
for spec in result.param_specs
    init    = isnothing(spec.initial) ? NaN32 : spec.initial
    trained = result.final_params[spec.name]
    @printf("%-28s  %10.4g  %10.4g  %+10.4g\n", spec.name, init, trained, trained - init)
end

@printf("\n%-6s  %8s  %8s  %8s\n", "flux", "target", "trained", "bias")
println("-" ^ 38)
for k in result.loss_config.flux_keys
    tgt  = result.loss_config.targets[k]
    val  = result.history[k][end]
    @printf("%-6s  %8.2f  %8.2f  %+8.2f\n", k, tgt, val, val - tgt)
end

figs = plot_training(result; save_dir=joinpath(@__DIR__, "output"))
display(figs.fig_loss)

# seasonal_cycle=true here too (this function's own default) -- now train-time and eval-time
# climate states finally match, unlike every prior run in this investigation.
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
display(cfigs.fig_summary)
