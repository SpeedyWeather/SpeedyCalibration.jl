# lrd-reweighting test at batch_days=2 -- see project_trenberth_lw_transmissivity_gradscale_fix
# memory for full history. The batch_days sweep is now complete and conclusive:
#
#   batch_days     grace period    best_smoothed_loss
#      30               0              1126.0
#      10              ~60              889.4
#       5              106               551.9
#       3              131               348.8
#       2              331 (true floor, extended run) 148.8
#
# At batch_days=2's true best point (batch 331, examples/output/
# trenberth_batchdays2_extended/), 5 of the 6 TRENBERTH_LOSS fluxes are
# essentially nailed: osr bias -1.2, sru +0.5, srd -1.4, lru -0.3. olr is
# +4.0 (a little off but fine). **lrd alone (bias +21.0) accounts for
# 0.3*21.0^2 ~= 132 of the total 148.8 loss -- ~89% of everything remaining.**
# This is the exact same flux that has grown monotonically since the very
# first per-parameter-trajectory analysis of the bd=10 baseline, regardless
# of batch_days -- a real, structural SW/LW coupling (fixing the shortwave
# fluxes physically warms/moistens the column in a way that pushes lrd up),
# not a training-loop or timescale artifact.
#
# FOLLOW-UP (2026-07-31): the lrd weight=1.0 test (examples/trenberth_bd2_lrdweight.jl)
# came back NEGATIVE -- best_smoothed_loss=496.9, more than 3x WORSE than the
# 0.3-weight baseline's 148.8. lrd's own bias barely moved (21.0 -> 19.0)
# while osr/sru/olr all degraded substantially (osr bias -1.2 -> -8.3). The
# lrd/other-flux trade-off is steep and inelastic -- 0.3->1.0 (>3x) badly
# overcorrected. This script tests a much milder step, lrd weight=0.5
# (between the 0.3 baseline and the 1.0 overcorrection), to see if there's a
# better point on the trade-off curve rather than assuming reweighting is a
# dead end entirely.
#
# Aliasing check unchanged: batch_days=2 -> batch_steps=72, samples_per_batch=10
# -> steps_per_sample=7, gcd(7,36)=1 -- clean.
#
# max_batches=400: matches the previous extended run's budget (which needed
# 361 batches to reach its floor and patience-stop); if reweighting changes
# the dynamics enough to need more, this may need extending in a follow-up.

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

# TRENBERTH_LOSS with lrd's weight raised 0.3 -> 0.5 (milder than the 1.0 test
# that overcorrected). Everything else identical to TRENBERTH_LOSS.
lrd_weighted_loss = LossConfig(
    [:osr, :sru, :srd, :olr, :lrd, :lru];
    targets = Dict(:osr => 101.9f0, :sru =>  23.0f0, :srd => 168.0f0,
                   :olr => 235.0f0, :lrd => 333.0f0, :lru => 398.0f0),
    weights = Dict(:osr => 1.0f0,   :sru =>  0.5f0,  :srd => 0.5f0,
                   :olr => 1.0f0,   :lrd =>  0.5f0,  :lru => 0.3f0),
)

result = calibrate!(
    param_specs,
    Optimisers.Adam(5f-3),
    lrd_weighted_loss,
    TrainingConfig(
        spinup_days       = 180,
        batch_days        = 2.0,
        samples_per_batch = 10,
        max_batches       = 400,
        grad_clip         = 5f0,
        trunc             = 31,
        nlayers           = 8,
        loss_threshold    = 1f-6,
        enable_lr_decay   = false,
        daily_cycle       = true,
    ),
    save_dir = joinpath(@__DIR__, "output/trenberth_bd2_lrdweight05"),
)

println(result)
println()
println("Convergence info:")
println("  stop_reason:        ", result.conv_info.stop_reason)
println("  best_batch:         ", result.conv_info.best_batch)
println("  best_smoothed_loss: ", round(result.conv_info.best_smoothed_loss, digits=2))
