# Diagnostic: does perturbing cloud_albedo alone (a pure-SW parameter) reliably
# move the longwave fluxes (OLR/LRD/LRU) away from their defaults, confirming a
# real cross-block physical coupling rather than a training-dynamics artifact?
#
# Context: examples/trenberth_gradscale_fix.jl (18-param run with tau0/tau0_pole/fl
# gradient-scaled down) showed cloud_albedo moving 0.60->0.71 while OLR/LRD/LRU
# drifted away from their (already-good) defaults. This script isolates the SW
# side cleanly: sweep cloud_albedo over a range with EVERY other parameter (incl.
# the Frierson LW-transmissivity block tau0_equator/tau0_pole/fl) held at its
# SpeedyWeather default, no training/AD involved at all -- just forward
# integration + the same DailyMeansCallback diagnostic used in run_climate_validation.
#
# Design: 180-day spinup + 60-day diagnostic window per sweep point, fresh model
# each time (no state carryover between sweep points), same start date as the
# training experiments (21 March, vernal equinox).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpeedyCalibration, SpeedyWeather, Statistics, Dates, Printf

# only cloud_albedo is ever set away from its SpeedyWeather default;
# param_specs just needs to exist so build_climate_sim knows the path.
param_specs = [
    ParamSpec(:cloud_albedo, [:shortwave_radiation, :clouds, :cloud_albedo];
        bounds=(0.25f0, 0.95f0)),
]

sweep = Float32[0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85]
spinup_days    = 180
diag_days      = 60
total_days     = spinup_days + diag_days

results = NamedTuple[]

for α in sweep
    t0 = time()
    # mirror SpeedyCalibration._run_clm's pattern: build model (no initialize!
    # yet), attach the callback, THEN initialize! -- add!-ing a callback after
    # initialize! is not the established pattern in this codebase.
    model, sg = SpeedyCalibration._build_model(
        Dict(:cloud_albedo => α), param_specs; trunc=31, nlayers=8)
    cb = DailyMeansCallback(sg)
    add!(model, :daily_means => cb)
    sim = initialize!(model)
    sim.variables.prognostic.clock.time = DateTime(2000, 3, 21)
    run!(sim; period=Day(total_days), output=false)

    idx = (spinup_days+1):total_days
    idx = intersect(idx, eachindex(cb.osr))
    stats = (
        cloud_albedo = α,
        osr = mean(cb.osr[idx]), sru = mean(cb.sru[idx]), srd = mean(cb.srd[idx]),
        olr = mean(cb.olr[idx]), lrd = mean(cb.lrd[idx]), lru = mean(cb.lru[idx]),
    )
    push!(results, stats)
    @printf("alpha=%.2f  osr=%6.2f  sru=%6.2f  srd=%6.2f  olr=%6.2f  lrd=%6.2f  lru=%6.2f  (%.1fs)\n",
            α, stats.osr, stats.sru, stats.srd, stats.olr, stats.lrd, stats.lru, time()-t0)
    flush(stdout)
end

println("\n=== SW->LW coupling sweep complete ===")
println("cloud_albedo, osr, sru, srd, olr, lrd, lru")
for r in results
    @printf("%.2f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
            r.cloud_albedo, r.osr, r.sru, r.srd, r.olr, r.lrd, r.lru)
end

# quick linear-regression slope (d flux / d cloud_albedo) for each flux, to
# quantify sensitivity magnitude and sign
xs = Float64[r.cloud_albedo for r in results]
xbar = mean(xs)
for k in (:osr, :sru, :srd, :olr, :lrd, :lru)
    ys = Float64[getfield(r, k) for r in results]
    ybar = mean(ys)
    slope = sum((xs .- xbar) .* (ys .- ybar)) / sum((xs .- xbar).^2)
    @printf("d(%s)/d(cloud_albedo) = %+7.2f  (W/m^2 per unit albedo)\n", k, slope)
end

mkpath(joinpath(@__DIR__, "output/sw_lw_coupling_diagnostic"))
open(joinpath(@__DIR__, "output/sw_lw_coupling_diagnostic/results.csv"), "w") do io
    println(io, "cloud_albedo,osr,sru,srd,olr,lrd,lru")
    for r in results
        println(io, "$(r.cloud_albedo),$(r.osr),$(r.sru),$(r.srd),$(r.olr),$(r.lrd),$(r.lru)")
    end
end
println("\nSaved to output/sw_lw_coupling_diagnostic/results.csv")
