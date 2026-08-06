# Real, full-budget batch_days=5 run under relative-error weighting -- never
# tested to convergence before now. Every prior bd=5 data point was either
# under the retired hand-tuned/default weighting (trenberth_batchdays5_sensitivity.jl)
# or the short 48-batch screening snapshot in
# examples/trenberth_batchdays_gradcount_sweep/ (part of a time-matched sweep
# vs bd=2/bd=10, not run to convergence).
#
# Why this specifically, not another bd=10 rerun: pulled the FULL, already-
# converged bd=10 relative-error run's history
# (output/trenberth_bd10_relerror/history.csv) and checked lrd's bias at the
# same ~240-simulated-day mark the screening sweep stopped at (batch 24):
# only +6.7 W/m^2, looking great -- but that run's lrd bias grows
# monotonically the entire rest of training, ending at +41.6 by convergence
# (worse than default). The screening sweep's "bd=10 fixes lrd" result is
# almost certainly the same early-training snapshot on a trajectory we
# already know gets much worse, not a real structural fix -- rerunning bd=10
# to convergence would almost certainly just reproduce that already-known
# bad result.
#
# bd=5 is the genuinely open question: never run to convergence under this
# weighting, and it was the standout on srd specifically (near zero bias) in
# the screening sweep, unlike bd=2 or bd=10. Does it share bd=10's monotonic
# lrd-drift problem, or does it actually stabilize at a better point? This
# run answers that for real, patience-based stop (not a fixed truncation).
#
# samples_per_batch=16: matches the "historical convention" value used
# throughout this project's bd=5 tests (steps_per_sample=180÷16=11,
# gcd(11,36)=1 -- clean, no aliasing), and was marginally the best of the
# three density variants tested in the screening sweep (14.99 vs 15.13/15.10
# mean|bias|, though that gap is small and not the deciding factor here).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

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

relerror_loss = LossConfig(
    [:osr, :sru, :srd, :olr, :lrd, :lru];
    targets = Dict(:osr => 101.9f0, :sru =>  23.0f0, :srd => 168.0f0,
                   :olr => 235.0f0, :lrd => 333.0f0, :lru => 398.0f0),
    weights = Dict(:osr => 1.00000f0, :sru => 19.62875f0, :srd => 0.36790f0,
                   :olr => 0.18802f0, :lrd => 0.09364f0,  :lru => 0.06555f0),
)

result = calibrate!(
    param_specs,
    Optimisers.Adam(5f-3),
    relerror_loss,
    TrainingConfig(
        spinup_days       = 180,
        batch_days        = 5.0,
        samples_per_batch = 16,
        max_batches       = 400,
        grad_clip         = 5f0,
        trunc             = 31,
        nlayers           = 8,
        loss_threshold    = 1f-6,
        enable_lr_decay   = false,
        daily_cycle       = true,
    ),
    save_dir = joinpath(@__DIR__, "..", "output", "trenberth_bd5_relerror"),
)

println(result)
println()
println("Convergence info:")
println("  stop_reason:        ", result.conv_info.stop_reason)
println("  best_batch:         ", result.conv_info.best_batch)
println("  best_smoothed_loss: ", round(result.conv_info.best_smoothed_loss, digits=2))
