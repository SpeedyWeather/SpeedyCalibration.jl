# Companion to sw_lw_coupling_diagnostic.jl: measure d(flux)/d(tau0_equator)
# directly (cloud_albedo and everything else pinned at SpeedyWeather defaults),
# so a grad_scale for the Frierson LW-transmissivity block can be chosen to
# match -- rather than exceed -- the rate at which cloud_albedo disrupts
# OLR/LRD/LRU (see project_trenberth_lw_transmissivity_gradscale_fix memory).
#
# Same method: 180-day spinup + 60-day diagnostic window, fresh model per point,
# vernal equinox start date, DailyMeansCallback diagnostics (no training/AD).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpeedyCalibration, SpeedyWeather, Statistics, Dates, Printf

param_specs = [
    ParamSpec(:tau0_equator, [:longwave_radiation, :transmissivity, :τ₀_equator];
        bounds=(2f0, 12f0)),
]

sweep = Float32[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
spinup_days = 180
diag_days   = 60
total_days  = spinup_days + diag_days

results = NamedTuple[]

for τ in sweep
    t0 = time()
    model, sg = SpeedyCalibration._build_model(
        Dict(:tau0_equator => τ), param_specs; trunc=31, nlayers=8)
    cb = DailyMeansCallback(sg)
    add!(model, :daily_means => cb)
    sim = initialize!(model)
    sim.variables.prognostic.clock.time = DateTime(2000, 3, 21)
    run!(sim; period=Day(total_days), output=false)

    idx = (spinup_days+1):total_days
    idx = intersect(idx, eachindex(cb.osr))
    stats = (
        tau0_equator = τ,
        osr = mean(cb.osr[idx]), sru = mean(cb.sru[idx]), srd = mean(cb.srd[idx]),
        olr = mean(cb.olr[idx]), lrd = mean(cb.lrd[idx]), lru = mean(cb.lru[idx]),
    )
    push!(results, stats)
    @printf("tau0_eq=%.1f  osr=%6.2f  sru=%6.2f  srd=%6.2f  olr=%6.2f  lrd=%6.2f  lru=%6.2f  (%.1fs)\n",
            τ, stats.osr, stats.sru, stats.srd, stats.olr, stats.lrd, stats.lru, time()-t0)
    flush(stdout)
end

println("\n=== tau0_equator sensitivity sweep complete ===")
println("tau0_equator, osr, sru, srd, olr, lrd, lru")
for r in results
    @printf("%.1f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
            r.tau0_equator, r.osr, r.sru, r.srd, r.olr, r.lrd, r.lru)
end

xs = Float64[r.tau0_equator for r in results]
xbar = mean(xs)
for k in (:osr, :sru, :srd, :olr, :lrd, :lru)
    ys = Float64[getfield(r, k) for r in results]
    ybar = mean(ys)
    slope = sum((xs .- xbar) .* (ys .- ybar)) / sum((xs .- xbar).^2)
    @printf("d(%s)/d(tau0_equator) = %+7.2f  (W/m^2 per unit tau0)\n", k, slope)
end

mkpath(joinpath(@__DIR__, "output/tau0_equator_sensitivity_sweep"))
open(joinpath(@__DIR__, "output/tau0_equator_sensitivity_sweep/results.csv"), "w") do io
    println(io, "tau0_equator,osr,sru,srd,olr,lrd,lru")
    for r in results
        println(io, "$(r.tau0_equator),$(r.osr),$(r.sru),$(r.srd),$(r.olr),$(r.lrd),$(r.lru)")
    end
end
println("\nSaved to output/tau0_equator_sensitivity_sweep/results.csv")
