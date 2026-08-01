# Third sweep in the SW<->LW coupling investigation: measure d(flux)/d(fl)
# directly (cloud_albedo, tau0_equator, tau0_pole pinned at SpeedyWeather
# defaults). fl had by far the largest raw AD gradient of the three LW
# transmissivity params (~600-617 vs tau0_equator's ~400) despite living in a
# much narrower [0,0.5] range -- this sweep tests whether that's because its
# physical sensitivity is genuinely large, and whether its sign helps or
# fights the correction tau0_equator provides.
#
# Same method as cloud_albedo/tau0_equator sweeps: 180d spinup + 60d diagnostic
# window, fresh model per point, vernal equinox start, no training/AD.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpeedyCalibration, SpeedyWeather, Statistics, Dates, Printf

param_specs = [
    ParamSpec(:fl, [:longwave_radiation, :transmissivity, :fₗ];
        bounds=(0f0, 0.5f0)),
]

sweep = Float32[0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
spinup_days = 180
diag_days   = 60
total_days  = spinup_days + diag_days

results = NamedTuple[]

for fl in sweep
    t0 = time()
    model, sg = SpeedyCalibration._build_model(
        Dict(:fl => fl), param_specs; trunc=31, nlayers=8)
    cb = DailyMeansCallback(sg)
    add!(model, :daily_means => cb)
    sim = initialize!(model)
    sim.variables.prognostic.clock.time = DateTime(2000, 3, 21)
    run!(sim; period=Day(total_days), output=false)

    idx = (spinup_days+1):total_days
    idx = intersect(idx, eachindex(cb.osr))
    stats = (
        fl = fl,
        osr = mean(cb.osr[idx]), sru = mean(cb.sru[idx]), srd = mean(cb.srd[idx]),
        olr = mean(cb.olr[idx]), lrd = mean(cb.lrd[idx]), lru = mean(cb.lru[idx]),
    )
    push!(results, stats)
    @printf("fl=%.2f  osr=%6.2f  sru=%6.2f  srd=%6.2f  olr=%6.2f  lrd=%6.2f  lru=%6.2f  (%.1fs)\n",
            fl, stats.osr, stats.sru, stats.srd, stats.olr, stats.lrd, stats.lru, time()-t0)
    flush(stdout)
end

println("\n=== fl sensitivity sweep complete ===")
println("fl, osr, sru, srd, olr, lrd, lru")
for r in results
    @printf("%.2f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
            r.fl, r.osr, r.sru, r.srd, r.olr, r.lrd, r.lru)
end

xs = Float64[r.fl for r in results]
xbar = mean(xs)
for k in (:osr, :sru, :srd, :olr, :lrd, :lru)
    ys = Float64[getfield(r, k) for r in results]
    ybar = mean(ys)
    slope = sum((xs .- xbar) .* (ys .- ybar)) / sum((xs .- xbar).^2)
    @printf("d(%s)/d(fl) = %+8.2f  (W/m^2 per unit fl)\n", k, slope)
end

mkpath(joinpath(@__DIR__, "output/fl_sensitivity_sweep"))
open(joinpath(@__DIR__, "output/fl_sensitivity_sweep/results.csv"), "w") do io
    println(io, "fl,osr,sru,srd,olr,lrd,lru")
    for r in results
        println(io, "$(r.fl),$(r.osr),$(r.sru),$(r.srd),$(r.olr),$(r.lrd),$(r.lru)")
    end
end
println("\nSaved to output/fl_sensitivity_sweep/results.csv")
