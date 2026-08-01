# batch_days=5 sensitivity test -- see project_trenberth_lw_transmissivity_gradscale_fix
# memory for full history. Two prior tests (batch_days=10 baseline vs. two
# batch_days=30 variants, one confounded by diurnal aliasing, one clean)
# established a clear, reproducible, *inverse* relationship between batch_days
# and the length of the early "grace period" before the joint 20-param
# TRENBERTH_LOSS optimization starts drifting monotonically worse:
#
#   batch_days=10 (baseline, 240 batches): ~60-batch grace period, then
#     monotonic worsening (lrd's bias grows the whole time but is masked
#     early by srd's larger, faster-shrinking initial bias).
#   batch_days=30 (v1 aliased AND v2 aliasing-corrected, both ~25-30
#     batches): ZERO grace period -- best batch is batch 1, every single
#     subsequent batch is worse, in both the confounded and clean version.
#
# This is the opposite of the original timescale-mismatch hypothesis's
# predicted direction (longer window -> better-equilibrated, truer gradient
# -> should help, not hurt). The revised mechanism: this is one continuously
# -run simulation with no state reset between batches, so a *longer* single
# batch window gives the slow LW/thermal feedback (cloud_albedo/fl/
# tau0_equator -> lrd, confirmed real physics in sw_lw_coupling_diagnostic.jl)
# more time to compound *within that one batch's own averaging window*,
# making it show up in the batch-mean gradient sooner, not later. Under this
# story, batch_days=10's grace period isn't "enough time to equilibrate" --
# it's just short enough that the slow feedback hasn't fully developed within
# any single batch yet, letting the fast SW correction dominate the visible
# signal for a while.
#
# This predicts, sharply and falsifiably: batch_days=5 (shorter than the
# baseline) should show a LONGER grace period than the baseline's ~60
# batches, not a shorter one. If the grace period instead shrinks or matches
# batch_days=10's, the "within-batch compounding" mechanism is wrong and a
# different explanation is needed (e.g. hypothesis 3, structural/no-joint-
# stationary-point).
#
# Aliasing check: batch_days=5 -> batch_steps=ceil(5*36)=180.
# samples_per_batch=16 -> steps_per_sample=180÷16=11, gcd(11,36)=1 --
# identical sampling resolution/diurnal coverage to the batch_days=10
# baseline (360÷32=11) and the batch_days=30 v2 test (1080÷98=11). All three
# batch_days points now share the exact same steps_per_sample=11, so
# batch_days is the only varied factor across all three runs.
#
# max_batches=150: half the AD-pass cost per batch of the baseline (16
# samples vs 32), so this budget costs about as much wall-time as the
# baseline's own 240-batch run despite covering 5x as many batches -- needed
# because if the grace-period-lengthens prediction is right, we may need well
# past 60 batches to see where (or whether) it eventually reverses.

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
        batch_days        = 5.0,
        samples_per_batch = 16,
        max_batches       = 150,  # default patience=30 can still fire well within this budget if
                                   # a reversal happens early; 150 gives ~2.5x the baseline's own
                                   # ~60-batch grace period as runway in case it lengthens as predicted.
        grad_clip         = 5f0,
        trunc             = 31,
        nlayers           = 8,
        loss_threshold    = 1f-6,
        enable_lr_decay   = false,
        daily_cycle       = true,
    ),
    save_dir = joinpath(@__DIR__, "output/trenberth_batchdays5_sensitivity"),
)

println(result)
println()
println("Convergence info:")
println("  stop_reason:        ", result.conv_info.stop_reason)
println("  best_batch:         ", result.conv_info.best_batch)
println("  best_smoothed_loss: ", round(result.conv_info.best_smoothed_loss, digits=2))
