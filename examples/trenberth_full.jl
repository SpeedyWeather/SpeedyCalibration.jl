# Reproduces the full 6-flux Trenberth calibration from the master thesis.
# Expected runtime: ~1 hour at trunc=31 (batch_days=2, see below).
#
# .jl twin of trenberth_full.ipynb -- keep both in sync. Current state matches
# the notebook's final production config (2026-08-01): batch_days=2,
# lrd weight=0.7. Full derivation in project_trenberth_lw_transmissivity_gradscale_fix
# memory. Summary of the path taken, chronologically:
#   1. grad_scale fix (below) stopped a catastrophic blowup, but training still
#      drifted worse after an early best point (batch_days=10: best 889.4@batch60,
#      then climbed for the rest of a 240-batch run).
#   2. Root cause: not post-optimum drift -- lrd's bias grew monotonically from
#      batch 1 the whole time, masked early by srd's larger, faster-shrinking bias.
#   3. A batch_days sweep (30/10/5/3/2) found batch_days is the dominant lever:
#      shorter is strictly better on every metric. batch_days=2 reaches a true
#      floor of best_smoothed_loss=148.8 (lrd weight=0.3), 5/6 fluxes essentially
#      nailed, lrd the one remaining gap (+21 W/m²).
#   4. lrd-reweighting (0.3 -> 0.7) closes most of that gap -- an initial
#      comparison looked like reweighting didn't help, but that compared
#      self-reported losses across different weight functions (invalid);
#      rescored under one fixed reference weighting, weight=0.7 is a real,
#      31%-better result at true 7-year climate equilibrium, not just the
#      training proxy.
#
# FIX (2026-07-30, see project_trenberth_lw_transmissivity_gradscale_fix memory):
# the original 23-param version of this script diverged (best loss 601@batch57,
# then climbed to 1902-2133 by batch 150/300 across two independent runs).
# Root-caused via a 6-stage parameter-count ablation + three single-parameter
# sensitivity sweeps (cloud_albedo, tau0_equator, fl; results in
# examples/output/{sw_lw_coupling_diagnostic,tau0_equator_sensitivity_sweep,
# fl_sensitivity_sweep}/results.csv). Two independent problems, both fixed below:
#
# 1. tau0_equator/tau0_pole/fl (Frierson LW transmissivity) never received a
#    grad_scale, unlike absorptivity_water_vapor (which needed one for the same
#    reason: different physical units/bounds than the mostly-fractional SW
#    block). Their raw AD gradients (measured ~400/~60/~600 respectively) then
#    dominated the shared grad_clip=5.0 L2-norm budget alongside cloud_albedo
#    (~400), and swung batch-to-batch in a fight against the SW-albedo block
#    (whose sensitivity sweep shows d(olr)/d(cloud_albedo)=-76.5 etc. -- a real,
#    physical, opposite-signed coupling, not a training artifact: fixing OSR via
#    cloud_albedo unavoidably cools the system and drags OLR/LRD/LRU down too).
#    grad_scale values below (0.04/0.25/0.025) were validated in
#    examples/trenberth_gradscale_fix.jl: they cut the divergence ratio
#    (final/best smoothed loss) from 2.97x to 1.25x -- stable, though the joint
#    18-param test run still showed the LW block *overcorrecting* past target
#    after ~150 batches (a residual multi-parameter tuning problem, not
#    instability -- see the memory entry for detail). emissivity_ocean/land
#    grad_scale below are a lower-confidence *reasoned* addition (not
#    independently ablation-tested the way tau0/fl were): emissivity_ocean's
#    raw gradient was ~75 in the one 300-batch run that included it, large
#    enough to plausibly cause the same grad_clip-budget competition.
# 2. infiltration_fraction, ocean_moisture, snow_melting_threshold had
#    bit-exact-zero AD gradient in every run that included them (confirmed via
#    `grad_<name>.values == 0.0` for all batches, not just "small") and were
#    REMOVED below rather than merely re-scaled. ocean_moisture is structural:
#    it's only read in SpeedyWeather's `initialize!`, before the online
#    single-timestep method's differentiated `timestep!` window even starts, so
#    no grad_scale/AD fix under this training method can help. infiltration_
#    fraction/snow_melting_threshold both route through the same
#    `launch!(...)`-packed `NamedTuple` kernel pattern already known to zero out
#    `lsc_rh_threshold` (see MEMORY.md "Parameter AD Limitations") -- likely an
#    Enzyme activity-propagation gap through that abstraction, not something
#    fixable from SpeedyCalibration.jl. (snow_melting_threshold already carried
#    a grad_scale=0.01 below prior to this fix -- that was scaling an exact
#    zero, i.e. had no effect; the real issue was never magnitude.)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpeedyCalibration, Optimisers, CairoMakie, Dates

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
    # Longwave transmissivity (Frierson scheme) -- grad_scale validated in
    # examples/trenberth_gradscale_fix.jl (divergence ratio 2.97x -> 1.25x)
    ParamSpec(:tau0_equator,
        [:longwave_radiation, :transmissivity, :τ₀_equator];
        bounds=(2f0, 12f0), initial=6f0, grad_scale=0.04f0),
    ParamSpec(:tau0_pole,
        [:longwave_radiation, :transmissivity, :τ₀_pole];
        bounds=(0.3f0, 4f0), initial=1.5f0, grad_scale=0.25f0),
    ParamSpec(:fl,
        [:longwave_radiation, :transmissivity, :fₗ];
        bounds=(0.0f0, 0.5f0), initial=0.1f0, grad_scale=0.025f0),
    # Longwave emissivity -- grad_scale is a reasoned addition (not
    # independently ablation-tested), see header comment
    ParamSpec(:emissivity_ocean,
        [:longwave_radiation, :radiative_transfer, :emissivity_ocean];
        bounds=(0.80f0, 1.00f0), initial=0.98f0, grad_scale=0.2f0),
    ParamSpec(:emissivity_land,
        [:longwave_radiation, :radiative_transfer, :emissivity_land];
        bounds=(0.80f0, 1.00f0), initial=0.98f0, grad_scale=0.5f0),
    # Land hydrology params (infiltration_fraction, ocean_moisture,
    # snow_melting_threshold) REMOVED: confirmed bit-exact-zero AD gradient in
    # every run, dead weight under the online single-timestep training method
    # (see header comment and project_trenberth_lw_transmissivity_gradscale_fix
    # memory for root cause).
]

# lrd weight corrected 0.3 -> 0.7 (2026-08-01): a finer reweighting sweep
# (0.3 through 1.0) found a clean local optimum at 0.7, confirmed at true
# climate equilibrium, not just the training proxy -- see header comment.
loss_config = LossConfig(
    [:osr, :sru, :srd, :olr, :lrd, :lru];
    targets = Dict(:osr => 101.9f0, :sru =>  23.0f0, :srd => 168.0f0,
                   :olr => 235.0f0, :lrd => 333.0f0, :lru => 398.0f0),
    weights = Dict(:osr => 1.0f0,   :sru =>  0.5f0,  :srd => 0.5f0,
                   :olr => 1.0f0,   :lrd =>  0.7f0,  :lru => 0.3f0),
)

# batch_days=2/samples_per_batch=10 (steps_per_sample=7, gcd(7,36)=1 -- clean
# diurnal coverage): the winning point of the batch_days sweep, see header
# comment. Resume-safe (matches Section 10's ablation pattern in the notebook):
# loads an existing result instead of re-training if one is already saved.
save_path = joinpath(@__DIR__, "output", "trenberth_full_result.jld2")
if isfile(save_path)
    result = load_result(save_path)
    println("Loaded existing result from: $save_path")
else
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
            daily_cycle       = true,
        ),
    )
    mkpath(dirname(save_path))
    save_result(result, save_path)
end

figs = plot_training(result; save_dir=joinpath(@__DIR__, "output/trenberth_full"))
display(figs.fig_loss)

# always validate result.best_params, not final_params -- training can drift
# past its optimum, see header comment.
clm = run_climate_validation(result; n_years=7, stat_years=5, dt=Minute(20))
cfigs = plot_climate(clm; save_dir=joinpath(@__DIR__, "output/trenberth_full"))
display(cfigs.fig_summary)
