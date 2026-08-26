# Full 6-flux Trenberth calibration, Frierson LW scheme (basic batched single-timestep
# gradients, same method as trenberth_full.ipynb) -- one new lever added: absorptivity_cloud_base.
#
# Context: the JeevanjeeRadiation LW-scheme swap (examples/trenberth_jeevanjee_full/) was piloted
# as a structural fix for the olr/lrd coupling, but the production-resolution spinup turned out
# unstable at both ends of the emissivity_atmosphere range tried (0.3 survives ~71 real days then
# NaNs from a slow cooling drift; 0.95 NaNs within ~9 days from a fast runaway-warming failure) --
# a real physics/numerics problem, not just a bad starting value. Per direction from the user
# (2026-08-25), parking that branch and returning to the basic single-step Frierson-scheme
# approach instead, adding the one untapped, already gradient-checked lever identified by the
# source-code audit in trenberth_investigation_writeup.ipynb (Section 6):
#
#   `absorptivity_cloud_base` (BackgroundShortwaveTransmissivity) -- SW *absorption* inside cloudy
#   layers, structurally different from cloud_albedo's *reflection*. Confirmed nonzero, real
#   gradient in a prior gradient-check run (mean|grad|=43.3, comparable to absorptivity_dry_air's
#   60.4, far below cloud_albedo's 980.8 -- should coexist fine in the shared grad_clip budget
#   without needing its own grad_scale). Its sibling `absorptivity_cloud_limit` is a confirmed
#   structural zero gradient (the `min(absorptivity_cloud_base*q_base, absorptivity_cloud_limit)`
#   cap in shortwave_transmissivity.jl -- differentiating through min() zeros the non-active
#   branch) and stays excluded.
#
# This targets `srd` specifically (the flux this lever affects), not `lrd` (still structurally
# coupled to `olr` through Frierson's shared transmissivity field -- unresolved, this run doesn't
# claim to fix it). Everything else -- SW/albedo block, existing LW-transmissivity block +
# grad_scale, relative-error flux weighting, batch_days=2 -- is identical to trenberth_full.ipynb's
# already-validated production config; only the one new parameter is added.
#
# Expected runtime: ~8-12 hours at T31 (same order as trenberth_full.ipynb).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using SpeedyCalibration, Optimisers, CairoMakie, GeoMakie, Dates, Printf

param_specs = [
    # ── SW cloud reflection (unchanged from trenberth_full.ipynb) ──────────────
    ParamSpec(:cloud_albedo,
        [:shortwave_radiation, :clouds, :cloud_albedo];
        bounds=(0.25f0, 0.95f0), initial=0.60f0),
    ParamSpec(:stratocumulus_cover_max,
        [:shortwave_radiation, :clouds, :stratocumulus_cover_max];
        bounds=(0.25f0, 0.95f0), initial=0.60f0),
    ParamSpec(:stratocumulus_albedo,
        [:shortwave_radiation, :clouds, :stratocumulus_albedo];
        bounds=(0.10f0, 0.90f0), initial=0.50f0),
    ParamSpec(:precipitation_weight,
        [:shortwave_radiation, :clouds, :precipitation_weight];
        bounds=(0.0f0, 0.8f0), initial=0.20f0),

    # ── SW atmospheric absorption (unchanged, plus the one new lever) ──────────
    ParamSpec(:absorptivity_water_vapor,
        [:shortwave_radiation, :transmissivity, :absorptivity_water_vapor];
        bounds=(60f0, 140f0), initial=75f0, grad_scale=0.01f0),
    ParamSpec(:absorptivity_dry_air,
        [:shortwave_radiation, :transmissivity, :absorptivity_dry_air];
        bounds=(0.005f0, 0.060f0), initial=0.03135f0),
    ParamSpec(:absorptivity_aerosol,
        [:shortwave_radiation, :transmissivity, :absorptivity_aerosol];
        bounds=(0.005f0, 0.060f0), initial=0.03135f0),
    ParamSpec(:ozone_absorption,
        [:shortwave_radiation, :radiative_transfer, :ozone_absorption];
        bounds=(0.002f0, 0.020f0), initial=0.01f0),
    # NEW: cloud-base SW absorption -- targets srd specifically. Default=10, Nonnegative in
    # source; cloud_absorptivity_term = min(absorptivity_cloud_base*q_base, absorptivity_
    # cloud_limit=0.14) saturates once the product nears 0.14, so bounds kept moderate (2-30)
    # rather than wide-open -- gradient-checked nonzero at init=10 (mean|grad|=43.3, moved to
    # 10.58 in one gradient-check run), i.e. not yet saturated at the default.
    ParamSpec(:absorptivity_cloud_base,
        [:shortwave_radiation, :transmissivity, :absorptivity_cloud_base];
        bounds=(2f0, 30f0), initial=10f0),

    # ── Surface albedo (unchanged) ──────────────────────────────────────────────
    ParamSpec(:albedo_land,
        [:albedo, :land, :albedo_land];
        bounds=(0.10f0, 0.70f0), initial=0.40f0),
    ParamSpec(:albedo_high_vegetation,
        [:albedo, :land, :albedo_high_vegetation];
        bounds=(0.04f0, 0.26f0), initial=0.15f0),
    ParamSpec(:albedo_low_vegetation,
        [:albedo, :land, :albedo_low_vegetation];
        bounds=(0.05f0, 0.35f0), initial=0.20f0),
    ParamSpec(:albedo_snow,
        [:albedo, :land, :albedo_snow];
        bounds=(0.15f0, 0.75f0), initial=0.40f0),
    ParamSpec(:snow_depth_scale,
        [:albedo, :land, :snow_depth_scale];
        bounds=(0.005f0, 0.20f0), initial=0.05f0),
    ParamSpec(:albedo_ocean,
        [:albedo, :ocean, :albedo_ocean];
        bounds=(0.02f0, 0.10f0), initial=0.06f0),
    ParamSpec(:albedo_ice,
        [:albedo, :ocean, :albedo_ice];
        bounds=(0.30f0, 0.90f0), initial=0.60f0),

    # ── Longwave transmissivity (Frierson scheme, unchanged) ───────────────────
    # grad_scale validated in examples/trenberth_blowup_fix/trenberth_gradscale_fix.jl --
    # without it these gradients (raw magnitude ~400-600) fight cloud_albedo for the
    # shared grad_clip budget and cause a catastrophic blowup. Do not remove.
    ParamSpec(:tau0_equator,
        [:longwave_radiation, :transmissivity, :τ₀_equator];
        bounds=(2f0, 12f0), initial=6f0, grad_scale=0.04f0),
    ParamSpec(:tau0_pole,
        [:longwave_radiation, :transmissivity, :τ₀_pole];
        bounds=(0.3f0, 4f0), initial=1.5f0, grad_scale=0.25f0),
    ParamSpec(:fl,
        [:longwave_radiation, :transmissivity, :fₗ];
        bounds=(0.0f0, 0.5f0), initial=0.1f0, grad_scale=0.025f0),

    # ── Longwave emissivity (unchanged) ─────────────────────────────────────────
    ParamSpec(:emissivity_ocean,
        [:longwave_radiation, :radiative_transfer, :emissivity_ocean];
        bounds=(0.80f0, 1.00f0), initial=0.98f0, grad_scale=0.2f0),
    ParamSpec(:emissivity_land,
        [:longwave_radiation, :radiative_transfer, :emissivity_land];
        bounds=(0.80f0, 1.00f0), initial=0.98f0, grad_scale=0.5f0),

    # ── Land hydrology params REMOVED ─────────────────────────────────────────
    # infiltration_fraction, ocean_moisture, snow_melting_threshold all had
    # bit-exact-zero AD gradient in every run (dead weight under the online
    # single-timestep training method) -- see project_trenberth_lw_transmissivity_
    # gradscale_fix memory for root cause. Do not re-add without re-verifying.
    #
    # absorptivity_cloud_limit REMOVED -- confirmed structural zero gradient (min()
    # non-active branch), see header comment.
]

println("$(length(param_specs)) trainable parameters defined.")

# Relative-error weighting -- identical objective-derived scheme as trenberth_full.ipynb.
loss_config = LossConfig(
    [:osr, :sru, :srd, :olr, :lrd, :lru];
    targets = Dict(:osr => 101.9f0, :sru =>  23.0f0, :srd => 168.0f0,
                   :olr => 235.0f0, :lrd => 333.0f0, :lru => 398.0f0),
    weights = Dict(:osr => 1.00000f0, :sru => 19.62875f0, :srd => 0.36790f0,
                   :olr => 0.18802f0, :lrd => 0.09364f0,  :lru => 0.06555f0),
)

println("Loss configuration: $(length(loss_config.flux_keys))-flux MSE")
println()
@printf("  %-6s  %8s  %10s\n", "flux", "target", "weight")
println("  " * "-" ^ 28)
for k in loss_config.flux_keys
    @printf("  %-6s  %6.1f W/m²  %10.5f\n", k, loss_config.targets[k], loss_config.weights[k])
end

# Quick smoke test -- verifies the new param path/bounds work end to end, flags a zero gradient.
result_test = calibrate!(
    param_specs, Optimisers.Adam(5f-3), loss_config, quick_test_config(),
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

# Full production run -- identical hyperparameters to trenberth_full.ipynb's already-validated
# batch_days=2 config; only param_specs changed (added absorptivity_cloud_base).
save_path = joinpath(@__DIR__, "output", "trenberth_full_cloudabs_result.jld2")

result = calibrate!(
    param_specs,
    Optimisers.Adam(5f-3),
    loss_config,
    TrainingConfig(
        spinup_days       = 180,
        batch_days        = 2.0,
        samples_per_batch = 10,
        max_batches       = 400,
        loss_threshold    = 1f-6,
        enable_lr_decay   = false,
        trunc             = 31,
        nlayers           = 8,
        daily_cycle       = true,   # required: physical diurnal cycle must stay on
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
