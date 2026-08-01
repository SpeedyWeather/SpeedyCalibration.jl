# batch_days=3 sensitivity test -- see project_trenberth_lw_transmissivity_gradscale_fix
# memory for full history. Three prior tests now establish a clean,
# reproducible, monotonic relationship between batch_days and both the
# length of the early "grace period" (batches before the joint 20-param
# TRENBERTH_LOSS optimization starts drifting worse) AND the quality of the
# best point reached before that drift:
#
#   batch_days=30 (aliasing-controlled): 0-batch grace period, best=batch 1.
#   batch_days=10 (baseline, 240 batches): ~60-batch grace period,
#     best_smoothed_loss=889.4.
#   batch_days=5  (150-batch budget): 106-batch grace period,
#     best_smoothed_loss=551.9 (38% BETTER than the bd=10 floor) -- shorter
#     window is strictly better on both axes, confirming the "within-batch
#     compounding" mechanism (this is one continuously-run simulation, so a
#     longer single batch window gives the slow LW/thermal feedback more
#     time to compound and dominate the batch-mean gradient *within* that one
#     batch, not just across many batches).
#
# This test pushes one step further (bd=3, between the confirmed-good bd=5
# and untested shorter territory) to see whether the trend keeps improving
# monotonically or starts to plateau/reverse -- e.g. if batch_days gets too
# short, the per-batch flux estimate could become too noisy/under-sampled in
# simulated-time to give a good gradient at all, which would show up as a
# WORSE floor or shorter grace period than bd=5's, unlike the bd=30->10->5
# trend so far.
#
# Aliasing check: batch_days=3 -> batch_steps=ceil(3*36)=108.
# samples_per_batch=14 -> steps_per_sample=108÷14=7, gcd(7,36)=1 -- clean,
# full diurnal coverage (not matched to the other runs' steps_per_sample=11
# exactly -- at this batch_days there's no small samples_per_batch that gives
# exactly 11 without also using a much larger, more expensive one -- but gcd=1
# is what actually matters to avoid the aliasing failure mode, not the exact
# interval).
#
# max_batches=200: bd=5 needed 136 batches to reach and pass its floor; if the
# trend continues the bd=3 grace period could run well past 106, so this
# gives comfortable headroom. Cost/batch is somewhat cheaper than bd=5's (14
# AD passes/batch vs 16), so 200 batches here is a similar order of
# wall-time to bd=5's 136-batch run.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpeedyCalibration, Optimisers, Dates

param_specs = [
    ParamSpec(:cloud_albedo, [:shortwave_radiation, :clouds, :cloud_albedo];
        bounds=(0.25f0, 0.95f0), initial=0.60f0),
    ParamSpec(:stratocumulus_cover_max, [:shortwave_radiation, :clouds, :stratocumulus_cover_max];
        bounds=(0.25f0, 0.95f0), initial=0.60f0),
    ParamSpec(:stratocumulus_albedo, [:shortwave_radiation, :clouds, :stratocumulus_albedo];
        bounds=(0.10f0, 0.90f0), initial=0.50f0),
    ParamSpec(:precipitation_weight, [:shortwave_radiation, :clouds, :precipitation_weight];
        bounds=(0.0f0, 0.8f0), initial=0.20f0),
    ParamSpec(:absorptivity_water_vapor, [:shortwave_radiation, :transmissivity, :absorptivity_water_vapor];
        bounds=(60f0, 140f0), initial=75f0, grad_scale=0.01f0),
    ParamSpec(:absorptivity_dry_air, [:shortwave_radiation, :transmissivity, :absorptivity_dry_air];
        bounds=(0.005f0, 0.060f0), initial=0.03135f0),
    ParamSpec(:absorptivity_aerosol, [:shortwave_radiation, :transmissivity, :absorptivity_aerosol];
        bounds=(0.005f0, 0.060f0), initial=0.03135f0),
    ParamSpec(:ozone_absorption, [:shortwave_radiation, :radiative_transfer, :ozone_absorption];
        bounds=(0.002f0, 0.020f0), initial=0.01f0),
    ParamSpec(:albedo_land, [:albedo, :land, :albedo_land];
        bounds=(0.10f0, 0.70f0), initial=0.40f0),
    ParamSpec(:albedo_high_vegetation, [:albedo, :land, :albedo_high_vegetation];
        bounds=(0.04f0, 0.26f0), initial=0.15f0),
    ParamSpec(:albedo_low_vegetation, [:albedo, :land, :albedo_low_vegetation];
        bounds=(0.05f0, 0.35f0), initial=0.20f0),
    ParamSpec(:albedo_snow, [:albedo, :land, :albedo_snow];
        bounds=(0.15f0, 0.75f0), initial=0.40f0),
    ParamSpec(:snow_depth_scale, [:albedo, :land, :snow_depth_scale];
        bounds=(0.005f0, 0.20f0), initial=0.05f0),
    ParamSpec(:albedo_ocean, [:albedo, :ocean, :albedo_ocean];
        bounds=(0.02f0, 0.10f0), initial=0.06f0),
    ParamSpec(:albedo_ice, [:albedo, :ocean, :albedo_ice];
        bounds=(0.30f0, 0.90f0), initial=0.60f0),
    ParamSpec(:tau0_equator, [:longwave_radiation, :transmissivity, :τ₀_equator];
        bounds=(2f0, 12f0), initial=6f0, grad_scale=0.04f0),
    ParamSpec(:tau0_pole, [:longwave_radiation, :transmissivity, :τ₀_pole];
        bounds=(0.3f0, 4f0), initial=1.5f0, grad_scale=0.25f0),
    ParamSpec(:fl, [:longwave_radiation, :transmissivity, :fₗ];
        bounds=(0.0f0, 0.5f0), initial=0.1f0, grad_scale=0.025f0),
    ParamSpec(:emissivity_ocean, [:longwave_radiation, :radiative_transfer, :emissivity_ocean];
        bounds=(0.80f0, 1.00f0), initial=0.98f0, grad_scale=0.2f0),
    ParamSpec(:emissivity_land, [:longwave_radiation, :radiative_transfer, :emissivity_land];
        bounds=(0.80f0, 1.00f0), initial=0.98f0, grad_scale=0.5f0),
]

result = calibrate!(
    param_specs,
    Optimisers.Adam(5f-3),
    TRENBERTH_LOSS,
    TrainingConfig(
        spinup_days       = 180,
        batch_days        = 3.0,
        samples_per_batch = 14,
        max_batches       = 200,  # bd=5's grace period ran to batch 106 (patience-stopped at 136);
                                   # 200 gives headroom in case the grace period extends further still.
        grad_clip         = 5f0,
        trunc             = 31,
        nlayers           = 8,
        loss_threshold    = 1f-6,
        enable_lr_decay   = false,
        daily_cycle       = true,
    ),
    save_dir = joinpath(@__DIR__, "output/trenberth_batchdays3_sensitivity"),
)

println(result)
println()
println("Convergence info:")
println("  stop_reason:        ", result.conv_info.stop_reason)
println("  best_batch:         ", result.conv_info.best_batch)
println("  best_smoothed_loss: ", round(result.conv_info.best_smoothed_loss, digits=2))
