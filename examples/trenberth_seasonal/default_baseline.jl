# A clean, longer-window default (untrained) climate baseline, run once and saved, rather than
# re-deriving "default" bias from whatever single 7-year run happens to accompany each training
# experiment. Motivation: trenberth_full_cloudabs's own default-vs-trained validation showed
# srd's default bias at +30.63, while an earlier run (trenberth_perblock_clip) logged srd's
# default bias at +9.56 for what should be the identical untrained model -- a ~21 W/m² swing that
# is almost certainly run-to-run chaotic weather variability in a 5-year stat window, not a real
# config difference. n_years=15/stat_years=10 here (vs. the usual 7/5) to average over more of
# that chaotic spread before treating this as "the" reference default.
#
# Uses the current 21-param param_specs (same set as trenberth_full_cloudabs.jl, including the
# new absorptivity_cloud_base) purely for build_climate_sim's signature -- default climate itself
# doesn't depend on which params are in param_specs, only on the model's own compiled-in defaults,
# so this number is directly comparable across every run in this investigation regardless of
# which params that run happened to train.
#
# seasonal_cycle=true (this function's own default, matching the thesis methodology) -- this run
# nails down a stable seasonal-cycle default baseline specifically, ahead of the training-time
# seasonal_cycle work in trenberth_seasonal_full.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using SpeedyCalibration, SpeedyWeather, Dates, Printf, Statistics, JLD2
import SpeedyCalibration: _build_model

const TARGETS = Dict(:osr => 101.9f0, :sru => 23.0f0, :srd => 168.0f0,
                      :olr => 235.0f0, :lrd => 333.0f0, :lru => 398.0f0)

# Same param_specs list as trenberth_full_cloudabs.jl (paths only matter here, not values --
# default climate uses the model's own compiled-in defaults regardless).
param_specs = [
    ParamSpec(:cloud_albedo, [:shortwave_radiation, :clouds, :cloud_albedo]; bounds=(0.25f0, 0.95f0)),
    ParamSpec(:stratocumulus_cover_max, [:shortwave_radiation, :clouds, :stratocumulus_cover_max]; bounds=(0.25f0, 0.95f0)),
    ParamSpec(:stratocumulus_albedo, [:shortwave_radiation, :clouds, :stratocumulus_albedo]; bounds=(0.10f0, 0.90f0)),
    ParamSpec(:precipitation_weight, [:shortwave_radiation, :clouds, :precipitation_weight]; bounds=(0.0f0, 0.8f0)),
    ParamSpec(:absorptivity_water_vapor, [:shortwave_radiation, :transmissivity, :absorptivity_water_vapor]; bounds=(60f0, 140f0)),
    ParamSpec(:absorptivity_dry_air, [:shortwave_radiation, :transmissivity, :absorptivity_dry_air]; bounds=(0.005f0, 0.060f0)),
    ParamSpec(:absorptivity_aerosol, [:shortwave_radiation, :transmissivity, :absorptivity_aerosol]; bounds=(0.005f0, 0.060f0)),
    ParamSpec(:ozone_absorption, [:shortwave_radiation, :radiative_transfer, :ozone_absorption]; bounds=(0.002f0, 0.020f0)),
    ParamSpec(:absorptivity_cloud_base, [:shortwave_radiation, :transmissivity, :absorptivity_cloud_base]; bounds=(2f0, 30f0)),
    ParamSpec(:albedo_land, [:albedo, :land, :albedo_land]; bounds=(0.10f0, 0.70f0)),
    ParamSpec(:albedo_high_vegetation, [:albedo, :land, :albedo_high_vegetation]; bounds=(0.04f0, 0.26f0)),
    ParamSpec(:albedo_low_vegetation, [:albedo, :land, :albedo_low_vegetation]; bounds=(0.05f0, 0.35f0)),
    ParamSpec(:albedo_snow, [:albedo, :land, :albedo_snow]; bounds=(0.15f0, 0.75f0)),
    ParamSpec(:snow_depth_scale, [:albedo, :land, :snow_depth_scale]; bounds=(0.005f0, 0.20f0)),
    ParamSpec(:albedo_ocean, [:albedo, :ocean, :albedo_ocean]; bounds=(0.02f0, 0.10f0)),
    ParamSpec(:albedo_ice, [:albedo, :ocean, :albedo_ice]; bounds=(0.30f0, 0.90f0)),
    ParamSpec(:tau0_equator, [:longwave_radiation, :transmissivity, :τ₀_equator]; bounds=(2f0, 12f0)),
    ParamSpec(:tau0_pole, [:longwave_radiation, :transmissivity, :τ₀_pole]; bounds=(0.3f0, 4f0)),
    ParamSpec(:fl, [:longwave_radiation, :transmissivity, :fₗ]; bounds=(0.0f0, 0.5f0)),
    ParamSpec(:emissivity_ocean, [:longwave_radiation, :radiative_transfer, :emissivity_ocean]; bounds=(0.80f0, 1.00f0)),
    ParamSpec(:emissivity_land, [:longwave_radiation, :radiative_transfer, :emissivity_land]; bounds=(0.80f0, 1.00f0)),
]

n_years, stat_years = 15, 10
println("Building default (untrained) climate sim, trunc=31/nlayers=8, seasonal_cycle=true...")
model, sg = _build_model(Dict{Symbol,Float32}(), param_specs; trunc=31, nlayers=8, seasonal_cycle=true)
cb = DailyMeansCallback(sg)
add!(model, :daily_means => cb)
sim = initialize!(model)
sim.variables.prognostic.clock.time = DateTime(2000, 3, 21)

println("Running $n_years years (stats over final $stat_years)...")
run!(sim; period=Day(n_years * 365), output=false)

# Matches _equilibrium_stats's own indexing convention exactly (validation.jl:112-116) so this
# baseline is directly comparable to any run_climate_validation() call's `.default` field.
year_start = n_years - stat_years + 1
i0  = searchsortedfirst(cb.days, Float64((year_start - 1) * 365))
i1  = searchsortedlast(cb.days,  Float64(n_years * 365))
idx = i0:i1

@printf("\n%-6s  %8s  %10s  %10s\n", "flux", "target", "mean", "bias")
println("-" ^ 40)
means = Dict{Symbol,Float32}()
for k in (:osr, :sru, :srd, :olr, :lrd, :lru)
    v = mean(getfield(cb, k)[idx])
    means[k] = v
    @printf("%-6s  %8.2f  %10.2f  %+10.2f\n", k, TARGETS[k], v, v - TARGETS[k])
end
@printf("\nPrecipitation: %.2f mm/day\n", mean(cb.precip_total[idx]))

save_path = joinpath(@__DIR__, "output", "default_baseline_seasonal.jld2")
mkpath(dirname(save_path))
jldsave(save_path; means=means, n_years=n_years, stat_years=stat_years, days=cb.days,
        osr=cb.osr, srd=cb.srd, sru=cb.sru, olr=cb.olr, lrd=cb.lrd, lru=cb.lru)
println("\nSaved to: $save_path")
