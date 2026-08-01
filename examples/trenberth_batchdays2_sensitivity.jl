# batch_days=2 sensitivity test -- see project_trenberth_lw_transmissivity_gradscale_fix
# memory for full history. Four prior data points now establish a clean,
# reproducible, monotonic relationship between batch_days and both the
# length of the early "grace period" (batches before the joint 20-param
# TRENBERTH_LOSS optimization starts drifting worse) AND the quality of the
# best point reached before that drift:
#
#   batch_days=30: 0-batch grace period, best_smoothed_loss=1126 (batch 1).
#   batch_days=10 (baseline, 240 batches): ~60-batch grace period,
#     best_smoothed_loss=889.4.
#   batch_days=5  (150-batch budget): 106-batch grace period,
#     best_smoothed_loss=551.9.
#   batch_days=3  (200-batch budget, 161 actual, 8870s~2.5hr): 131-batch
#     grace period, best_smoothed_loss=348.8. srd landed almost exactly on
#     target (168.0) at the best point; osr/olr both made real progress too.
#     lrd is still the one flux that grows monotonically the whole run (same
#     mechanism as ever), just now a smaller fraction of a much smaller total.
#
# The trend has NOT plateaued through 4 points and cost is escalating fast
# (73min -> 148min from bd=5 to bd=3 alone). This test pushes one step
# further to see whether bd=2 keeps the trend going or starts to flatten out,
# before deciding whether pushing to bd=1 or committing to a value for the
# real 300-batch production run is the better use of remaining compute.
#
# Aliasing check: batch_days=2 -> batch_steps=ceil(2*36)=72.
# samples_per_batch=10 -> steps_per_sample=72÷10=7, gcd(7,36)=1 -- clean, and
# matches bd=3's steps_per_sample=7 exactly (same sampling density/diurnal
# coverage quality as the bd=3 test, just over a shorter window).
#
# max_batches=220: bd=3's grace period (131) plus patience(30) plus margin;
# if the trend continues similarly bd=2's grace period could land somewhere
# in the 140-170 range, this leaves headroom for patience to fire cleanly.

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
        batch_days        = 2.0,
        samples_per_batch = 10,
        max_batches       = 220,
        grad_clip         = 5f0,
        trunc             = 31,
        nlayers           = 8,
        loss_threshold    = 1f-6,
        enable_lr_decay   = false,
        daily_cycle       = true,
    ),
    save_dir = joinpath(@__DIR__, "output/trenberth_batchdays2_sensitivity"),
)

println(result)
println()
println("Convergence info:")
println("  stop_reason:        ", result.conv_info.stop_reason)
println("  best_batch:         ", result.conv_info.best_batch)
println("  best_smoothed_loss: ", round(result.conv_info.best_smoothed_loss, digits=2))
