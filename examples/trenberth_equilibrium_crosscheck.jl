# Group A of the 2026-08-01 plan (see project_trenberth_lw_transmissivity_gradscale_fix memory,
# "CORRECTION 2026-08-01" section): before trusting the newly-corrected finding that lrd
# weight=0.5 beats the 0.3 production baseline (under a fixed, apples-to-apples yardstick), we
# need to check whether that holds at TRUE 7-year climate equilibrium, not just the training
# batch-mean -- the 0.3 baseline already showed training and equilibrium can disagree (srd looked
# perfect in training but was worse than default at equilibrium).
#
# Two independent checks bundled into one script since neither depends on the other:
#
# A2: climate-validate the already-trained lrd-weight=0.5 run
#     (output/trenberth_bd2_lrdweight05/result.jld2) using the now-fixed run_climate_validation
#     (best_params, not final_params).
#
# A3: cross-check whether batch_days=2 itself was the right call at equilibrium, not just on the
#     training metric. The ORIGINAL batch_days=10 run's result.jld2 was overwritten earlier this
#     session (copied the bd=2 production result over it), but its full per-batch history
#     (including every trained parameter's value) is preserved in
#     output/trenberth_full/history.csv. best_params at its best_batch=60 is reconstructed here
#     directly from that CSV row -- no retraining needed -- and climate-validated the same way.
#
# Both use dt=Minute(20) and n_years=7/stat_years=5, matching trenberth_full_downstream.jl's
# validation of the bd=2 production run, so all three are directly comparable.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpeedyCalibration, CairoMakie, Dates, Printf, DelimitedFiles

targets = Dict(:osr=>101.9f0, :sru=>23.0f0, :srd=>168.0f0, :olr=>235.0f0, :lrd=>333.0f0, :lru=>398.0f0)
flux_keys = [:osr, :sru, :srd, :olr, :lrd, :lru]

function print_bias_table(label, clm)
    println("\n=== $label ===")
    @printf("%-6s  %8s  %9s  %9s  %9s  %9s\n",
            "flux", "target", "def val", "def bias", "trn val", "trn bias")
    println("-" ^ 60)
    for k in flux_keys
        tgt   = targets[k]
        d_val = getproperty(clm.default, k)
        t_val = getproperty(clm.trained, k)
        @printf("%-6s  %8.2f  %9.2f  %+9.2f  %9.2f  %+9.2f\n",
                k, tgt, d_val, d_val - tgt, t_val, t_val - tgt)
    end
    @printf("Precipitation:  default = %.2f mm/day  trained = %.2f mm/day  (ERA5 ≈ 2.74)\n",
            clm.default.precip_total, clm.trained.precip_total)
end

# ── A2: lrd weight=0.5 run ───────────────────────────────────────────────────
println("="^70)
println("A2: climate-validating lrd weight=0.5 run (output/trenberth_bd2_lrdweight05)")
println("="^70)
r05 = load_result(joinpath(@__DIR__, "output", "trenberth_bd2_lrdweight05", "result.jld2"))
println("best_batch=", r05.conv_info.best_batch, "  best_smoothed_loss=", round(r05.conv_info.best_smoothed_loss, digits=2))
clm05 = run_climate_validation(r05; n_years=7, stat_years=5, dt=Minute(20))
print_bias_table("lrd weight=0.5 -- climate equilibrium", clm05)

# ── A3: reconstruct batch_days=10 best_batch=60 checkpoint from CSV ────────
println("\n" * "="^70)
println("A3: reconstructing batch_days=10 best_batch=60 from history.csv, climate-validating")
println("="^70)

csv_path = joinpath(@__DIR__, "output", "trenberth_full", "history.csv")
raw = readdlm(csv_path, ',', header=true)
data, header = raw
header = vec(header)
col(name) = findfirst(==(name), header)
batch_col = col("batch")
row60 = findfirst(==(60), Int.(data[:, batch_col]))
isnothing(row60) && error("batch=60 not found in $csv_path")

param_names = [:cloud_albedo, :stratocumulus_cover_max, :stratocumulus_albedo, :precipitation_weight,
               :absorptivity_water_vapor, :absorptivity_dry_air, :absorptivity_aerosol, :ozone_absorption,
               :albedo_land, :albedo_high_vegetation, :albedo_low_vegetation, :albedo_snow,
               :snow_depth_scale, :albedo_ocean, :albedo_ice,
               :tau0_equator, :tau0_pole, :fl, :emissivity_ocean, :emissivity_land]

best_params_bd10 = Dict{Symbol,Float32}(
    p => Float32(data[row60, col(String(p))]) for p in param_names
)
println("Reconstructed best_params (batch=60):")
for p in param_names
    @printf("  %-28s = %.4g\n", p, best_params_bd10[p])
end

# param_specs: identical list to trenberth_full.ipynb / all sweep scripts (bounds/grad_scale don't
# affect climate validation -- only the trained values and paths matter for building the model).
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

# Synthetic TrainingResult: run_climate_validation only reads .config (trunc/nlayers/start_date),
# .param_specs, and .best_params -- history/final_params/conv_info/loss_config are unused by it.
fake_result = TrainingResult(
    Dict{Symbol,Vector}(),
    Dict{Symbol,Float32}(),
    best_params_bd10,
    (best_batch=60, best_smoothed_loss=889.44f0, total_batches=240, total_time=0.0, stop_reason="reconstructed from CSV"),
    TrainingConfig(trunc=31, nlayers=8),
    TRENBERTH_LOSS,
    param_specs,
)

clm_bd10 = run_climate_validation(fake_result; n_years=7, stat_years=5, dt=Minute(20))
print_bias_table("batch_days=10 (original), best_batch=60 -- climate equilibrium", clm_bd10)

println("\n" * "="^70)
println("Reference: batch_days=2, lrd weight=0.3 (current production config) already validated")
println("in trenberth_full_downstream.jl / output/trenberth_full_downstream.log:")
println("  osr -0.25  sru -3.80  srd -13.91  olr +12.57  lrd +30.01  lru +3.35")
println("="^70)
println("\nDone.")
