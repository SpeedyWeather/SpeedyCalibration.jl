# Tests hypothesis 1 from project_trenberth_lw_transmissivity_gradscale_fix
# memory (timescale mismatch: LW/thermal fields need longer than batch_days=10
# to re-equilibrate between gradient samples) against the persistent post-fix
# drift found in trenberth_full.jl/.ipynb's 240-batch run.
#
# New evidence motivating this specific test (2026-07-30, from re-analysis of
# the already-completed output/trenberth_full_result.jld2, no new compute):
# grad_fl, grad_tau0_equator, grad_tau0_pole, grad_cloud_albedo have the SAME
# SIGN in every single one of the 240 batches (0 sign flips) -- this is not
# noisy oscillation around a stationary point, it's a persistently one-directional
# push for the entire run. The weighted-loss "best point" at batch 60 is an
# artifact of srd's bias crossing zero around batch 60-75 (it was the largest
# initial residual, -609 contribution -> ~0), not a real turning point in
# parameter space: every tracked parameter keeps moving the same direction
# before AND after batch 60. Once srd's cheap win is exhausted, lrd's bias
# (which grows monotonically from batch 1, weighted contribution 0.7 -> 548.3
# by batch 240, ending up >48% of total loss) dominates and the total starts
# rising. Also: the loss reversal (~batch 60-75) happens entirely before the
# first LR decay (batch 110, still flat at 5e-3) -- already rules out an
# LR-decay-schedule interaction as the trigger without needing a separate test.
#
# This script folds that ruled-out LR-decay question and the batch_days
# hypothesis into ONE run: enable_lr_decay=false (flat 5e-3 the whole time,
# closes the loop on the LR question for certain) + batch_days=30 (vs the
# baseline's 10) to test whether letting the slow (temperature/humidity)
# fields equilibrate longer between batches changes the lrd trajectory.
#
# 100 batches (not 300): at 3x the simulated days per batch of the baseline,
# this already covers ~3x the total simulated days of the baseline's own
# batch-60 turning point, which is enough runway to see whether the
# monotonic lrd growth persists, slows, or reverses. Kept short deliberately
# per the "smallest experiment that answers the question" rule -- a full
# 300-batch run at batch_days=30 would cost ~3x the wall-time of the already-
# expensive baseline for marginal extra signal.

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
        batch_days        = 30.0,
        samples_per_batch = 32,
        max_batches       = 100,
        grad_clip         = 5f0,
        trunc             = 31,
        nlayers           = 8,
        loss_threshold    = 1f-6,
        enable_lr_decay   = false,
        daily_cycle       = true,
    ),
    save_dir = joinpath(@__DIR__, "output/trenberth_batchdays30_sensitivity"),
)

println(result)
println()
println("Convergence info:")
println("  stop_reason:        ", result.conv_info.stop_reason)
println("  best_batch:         ", result.conv_info.best_batch)
println("  best_smoothed_loss: ", round(result.conv_info.best_smoothed_loss, digits=2))
