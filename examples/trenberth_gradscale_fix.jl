# Diagnostic experiment: does rescaling the Frierson longwave-transmissivity
# gradients (τ₀_equator, τ₀_pole, fₗ) restore stability?
#
# Context: examples/output/trenberth_ablation/stage{3,4}_n{15,18} showed a clean
# divergence onset exactly when tau0_equator/tau0_pole/fl are added (n15 stable,
# best_loss=357; n18 diverges, best_loss=640@batch53 -> 1902@batch150).
#
# Per-parameter mean|grad| in stage4_n18 (raw gradient fed to the optimizer,
# i.e. after grad_scale and the sigmoid chain-rule factor -- see training.jl):
#   fl            617   (NO grad_scale set -- bug/oversight)
#   cloud_albedo  407
#   tau0_equator  402   (NO grad_scale set)
#   tau0_pole      58   (NO grad_scale set)
#   -- next largest SW param: absorptivity_dry_air ~40, most others 1-35 --
# Compare: absorptivity_water_vapor already carries grad_scale=0.01 in
# trenberth_full.jl specifically because its physical units (60-140) don't match
# the mostly-fractional SW block. tau0_equator/tau0_pole/fl never got the same
# treatment when they were added, and now dominate/compete with cloud_albedo for
# the tiny grad_clip=5.0 L2-norm budget (grad_clip is essentially always active:
# raw grad norms are ~700-850, clip is 5.0), which is not itself pathological
# with ONE dominant large param (stage3: cloud_albedo alone, stable) but becomes
# a poorly-conditioned multi-way competition with THREE comparable-magnitude,
# partly-opposed-sign large gradients (stage4).
#
# grad_scale values below target ~15 (matching the well-behaved SW-block median)
# for each param, computed as target / stage4_n18_mean_grad:
#   fl:            15/617 ≈ 0.024 -> 0.025
#   tau0_equator:  15/402 ≈ 0.037 -> 0.04
#   tau0_pole:     15/58  ≈ 0.26  -> 0.25
#
# Same 18 params, same TrainingConfig as stage4_n18 (spinup=180d, batch_days=10,
# samples_per_batch=32, max_batches=150, grad_clip=5.0, trunc=31, nlayers=8),
# so the comparison is apples-to-apples against the saved stage4_n18 result.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpeedyCalibration, Optimisers, Dates

param_specs = [
    # SW cloud reflection
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
    # SW atmospheric absorption
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
    # Surface albedo
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
    # Longwave transmissivity (Frierson scheme) -- grad_scale added (see header)
    ParamSpec(:tau0_equator,
        [:longwave_radiation, :transmissivity, :τ₀_equator];
        bounds=(2f0, 12f0), initial=6f0, grad_scale=0.04f0),
    ParamSpec(:tau0_pole,
        [:longwave_radiation, :transmissivity, :τ₀_pole];
        bounds=(0.3f0, 4f0), initial=1.5f0, grad_scale=0.25f0),
    ParamSpec(:fl,
        [:longwave_radiation, :transmissivity, :fₗ];
        bounds=(0.0f0, 0.5f0), initial=0.1f0, grad_scale=0.025f0),
]

result = calibrate!(
    param_specs,
    Optimisers.Adam(5f-3),
    TRENBERTH_LOSS,
    TrainingConfig(
        spinup_days    = 180,
        batch_days     = 10.0,
        samples_per_batch = 32,
        max_batches    = 150,
        grad_clip      = 5f0,
        trunc          = 31,
        nlayers        = 8,
        loss_threshold = 1f-6,   # disable early "converged" stop, mirror stage4_n18
    ),
    save_dir = joinpath(@__DIR__, "output/trenberth_gradscale_fix"),
)

println(result)
