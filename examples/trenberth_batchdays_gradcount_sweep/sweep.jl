# Rigorous joint sweep of batch_days x samples_per_batch.
#
# Every previous batch_days test (examples/trenberth_batchdays_sweep/) scaled
# samples_per_batch UP alongside batch_days, chosen only to satisfy the
# diurnal-aliasing constraint gcd(steps_per_sample, steps_per_day=36)=1 while
# holding steps_per_sample roughly fixed. That means "batch_days" was never
# tested as an isolated factor -- every comparison also changed how many
# gradient samples got averaged into one batch-mean gradient, which mostly
# controls estimator NOISE/VARIANCE, not the window-length effect the
# "instantaneous vs equilibrium sensitivity" hypothesis is actually about.
# This sweep varies both independently so the two effects can be told apart.
#
# GRID: batch_days in {2, 5, 10}, at each testing 2-3 alias-safe
# (gcd(steps_per_sample,36)=1) samples_per_batch values spanning a real
# density range (samples/day). Valid (spb, steps_per_sample) pairs were
# enumerated programmatically -- see conversation 2026-08-05. batch_days=3
# skipped (too close to 2 to add much), batch_days=30 skipped (uniformly
# worst in every past test, most expensive per batch, low priority for a
# first pass -- can add later if this sweep's trend points that direction).
#
# TWO-STAGE DESIGN (same discipline as the original batch_days sensitivity
# sweep): this is a SCREENING pass, not a final answer. Short training
# budget and a shorter equilibrium validation (SCREEN_N_YEARS/
# SCREEN_STAT_YEARS, not the full 7yr/5yr standard) at every grid point, to
# find the promising region cheaply. Whichever point(s) look best get the
# full n_years=7/stat_years=5 validation afterward, as a separate follow-up
# script -- do not treat this sweep's own numbers as the final word on any
# single config.
#
# FIX (2026-08-05): the first version of this sweep used the same
# max_batches for every batch_days value. Since total simulated time
# trained = batch_days * batches_trained, that gave batch_days=10 rows ~5x
# more simulated days than batch_days=2 rows at the "same" budget (1160-1200
# vs 240 days) -- bd=10 looking better in that version could just as well
# have been "got 5x more training exposure" rather than batch_days actually
# winning. max_batches is now set PER batch_days value so that
# batch_days*max_batches is held constant at TOTAL_SIMULATED_DAYS across the
# whole grid -- an apples-to-apples comparison of "same total training
# budget, spent as few-long vs many-short batches." Chosen to equal what
# the batch_days=2 rows already used (120*2=240 days), so those two rows
# didn't need re-running; only the batch_days=5/10 rows were retrained at
# their corrected (smaller) budgets.
#
# Same 20-param set + relative-error weighting (current production loss,
# trenberth_full.ipynb Section 3) throughout -- only batch_days/
# samples_per_batch vary across the grid.
#
# Resume-safe per grid point (checks for an existing result.jld2 before
# training) and writes a running summary CSV after every point, by hand
# (matching the existing history.csv pattern in src/training.jl) rather than
# adding a CSV.jl/DataFrames.jl dependency this package doesn't have.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using SpeedyCalibration, Optimisers, Dates, Printf, Statistics

const GRID = [
    (batch_days=2.0,  samples_per_batch=10, label="bd2_spb10"),   # 5.0 samples/day (baseline density)
    (batch_days=2.0,  samples_per_batch=13, label="bd2_spb13"),   # 6.5 samples/day
    (batch_days=5.0,  samples_per_batch=13, label="bd5_spb13"),   # 2.6 samples/day
    (batch_days=5.0,  samples_per_batch=16, label="bd5_spb16"),   # 3.2 samples/day (matches historical convention)
    (batch_days=5.0,  samples_per_batch=31, label="bd5_spb31"),   # 6.2 samples/day
    (batch_days=10.0, samples_per_batch=14, label="bd10_spb14"),  # 1.4 samples/day
    (batch_days=10.0, samples_per_batch=31, label="bd10_spb31"),  # 3.1 samples/day (matches historical convention)
    (batch_days=10.0, samples_per_batch=61, label="bd10_spb61"),  # 6.1 samples/day
]

const TOTAL_SIMULATED_DAYS = 240   # = the batch_days=2 rows' original 120*2
const SCREEN_N_YEARS       = 3
const SCREEN_STAT_YEARS    = 1

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

sweep_dir = joinpath(@__DIR__, "..", "output", "trenberth_batchdays_gradcount_sweep")
mkpath(sweep_dir)
summary_path = joinpath(sweep_dir, "summary.csv")

header = "label,batch_days,samples_per_batch,samples_per_day,best_batch,best_smoothed_loss," *
         "osr_bias,sru_bias,srd_bias,olr_bias,lrd_bias,lru_bias,mean_abs_bias\n"
open(summary_path, "w") do io
    write(io, header)
end

rows_printed = String[]

for g in GRID
    max_batches = round(Int, TOTAL_SIMULATED_DAYS / g.batch_days)

    println("\n", "=" ^ 70)
    println("GRID POINT: ", g.label, "  batch_days=", g.batch_days,
            "  samples_per_batch=", g.samples_per_batch,
            "  max_batches=", max_batches, " (", g.batch_days * max_batches, " simulated days)")
    println("=" ^ 70)

    point_dir  = joinpath(sweep_dir, g.label)
    save_path  = joinpath(point_dir, "result.jld2")

    if isfile(save_path)
        result = load_result(save_path)
        println("Loaded existing result for ", g.label,
                "  best_batch=", result.conv_info.best_batch)
    else
        result = calibrate!(
            param_specs,
            Optimisers.Adam(5f-3),
            relerror_loss,
            TrainingConfig(
                spinup_days       = 180,
                batch_days        = g.batch_days,
                samples_per_batch = g.samples_per_batch,
                max_batches       = max_batches,
                grad_clip         = 5f0,
                trunc             = 31,
                nlayers           = 8,
                loss_threshold    = 1f-6,
                enable_lr_decay   = false,
                daily_cycle       = true,
            ),
            save_dir = point_dir,
        )
    end

    clm = run_climate_validation(result; n_years=SCREEN_N_YEARS, stat_years=SCREEN_STAT_YEARS, dt=Minute(20))

    biases = Dict{Symbol,Float32}()
    for k in result.loss_config.flux_keys
        tgt = result.loss_config.targets[k]
        biases[k] = getproperty(clm.trained, k) - tgt
    end
    mean_abs_bias = mean(abs.(collect(values(biases))))

    row = @sprintf("%s,%.1f,%d,%.2f,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
        g.label, g.batch_days, g.samples_per_batch, g.samples_per_batch / g.batch_days,
        result.conv_info.best_batch, result.conv_info.best_smoothed_loss,
        biases[:osr], biases[:sru], biases[:srd], biases[:olr], biases[:lrd], biases[:lru],
        mean_abs_bias)

    open(summary_path, "a") do io
        write(io, row)
    end
    push!(rows_printed, row)

    @printf("Result: best_batch=%d  best_loss=%.2f  mean|bias|=%.2f (screening: %dyr validation)\n",
            result.conv_info.best_batch, result.conv_info.best_smoothed_loss, mean_abs_bias, SCREEN_N_YEARS)
    println("Progress saved to ", summary_path)
end

println("\n\n", "=" ^ 90)
println("FULL SWEEP SUMMARY (screening budget: ", TOTAL_SIMULATED_DAYS,
        " simulated days per grid point, validation n_years=", SCREEN_N_YEARS,
        "/stat_years=", SCREEN_STAT_YEARS, ")")
println("=" ^ 90)
print(header)
for r in rows_printed
    print(r)
end
println("\nDone. Full CSV at: ", summary_path)
println("NOTE: this is a screening pass, not the final word -- whichever point(s) look best")
println("should get a full n_years=7/stat_years=5 validation before drawing conclusions.")
