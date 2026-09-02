# Tests the "slow-timescale coupling" hypothesis from
# project_trenberth_lw_transmissivity_gradscale_fix.md: does the ratio
# d(lrd)/d(theta) / d(olr)/d(theta) for the Frierson longwave transmissivity
# parameters (tau0_equator, tau0_pole, fl) change between a single-timestep
# gradient (compute_gradients!) and an N=20-step checkpointed gradient
# (compute_gradients_checkpointed!) evaluated at the SAME spun-up state?
#
# If the ratio is essentially unchanged, the olr/lrd coupling is already fully
# present instantaneously and is not a slow-timescale artifact that a longer
# gradient horizon could exploit. If the ratio changes substantially, that is
# evidence the single-timestep training method is blind to something a
# longer-horizon gradient could see.
#
# NOTE on what N=20 actually means physically: read off after the run below,
# since it depends on the model's Δt (printed). At T31/L8 the timestep is on
# the order of 30-40 min, so N=20 is on the order of 10-14 hours — well under
# one full day, let alone the multi-day soil/humidity feedback the original
# hypothesis is about. See the report for the honest caveat.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using SpeedyCalibration, SpeedyWeather, Dates, Printf, JLD2

# ─────────────────────────────────────────────────────────────────────────
# 1) Build model, apply Frierson longwave scheme + defaults, spin up.
#    Same conventions as checkpointed_calibration.ipynb / trenberth_full.jl:
#    T31/L8, daily_cycle=true, seasonal_cycle=false, start 2000-03-21,
#    spinup_days=20 (same as thesis_15param_shortwave.ipynb / the
#    checkpointed notebook -- NOT the 180-day calibrate! spinup, since this
#    is a diagnostic probe at one representative state, not a training run).
# ─────────────────────────────────────────────────────────────────────────

sg     = SpectralGrid(trunc=31, nlayers=8)
planet = Earth(sg; daily_cycle=true, seasonal_cycle=false)
model  = PrimitiveWetModel(sg; planet=planet)
# Default longwave_radiation = OneBandLongwave, whose default transmissivity component
# is FriersonLongwaveTransmissivity (see SpeedyWeather src/parameterizations/radiation/
# longwave_radiation.jl) -- same as trenberth_full.jl, which relies on this default
# rather than constructing it explicitly.

sim = initialize!(model)
sim.variables.prognostic.clock.time = DateTime(2000, 3, 21)
SpeedyWeather.initialize!(sim; period=Day(365*100), output=false)

spinup_days = 20
clock = sim.variables.prognostic.clock
steps_per_day = ceil(Int, Millisecond(Day(1)).value / Millisecond(clock.Δt).value)

println("Δt = $(clock.Δt), steps_per_day = $steps_per_day")
println("Spinning up $spinup_days days...")
t0 = time()
for _ in 1:(spinup_days * steps_per_day)
    SpeedyWeather.time_step!(sim)
end
@printf("Spinup complete in %.1f s.\n", time() - t0)

N = 20
@printf("N=%d checkpointed steps ≈ %.2f hours ≈ %.3f days of real elapsed time.\n",
        N, N * Millisecond(clock.Δt).value / 1000 / 3600, N * Millisecond(clock.Δt).value / 1000 / 86400)

# ─────────────────────────────────────────────────────────────────────────
# 2) Parameters under test: the three Frierson longwave transmissivity
#    params already implicated in the olr/lrd coupling
#    (project_trenberth_lw_transmissivity_gradscale_fix.md, "confirmed
#    mechanism #2"). Paths/bounds copied verbatim from trenberth_full.jl.
# ─────────────────────────────────────────────────────────────────────────

param_specs = [
    ParamSpec(:tau0_equator, [:longwave_radiation, :transmissivity, :τ₀_equator];
        bounds=(2f0, 12f0), initial=6f0),
    ParamSpec(:tau0_pole, [:longwave_radiation, :transmissivity, :τ₀_pole];
        bounds=(0.3f0, 4f0), initial=1.5f0),
    ParamSpec(:fl, [:longwave_radiation, :transmissivity, :fₗ];
        bounds=(0.0f0, 0.5f0), initial=0.1f0),
]

# ─────────────────────────────────────────────────────────────────────────
# Raw-sensitivity trick: build a single-flux LossConfig whose target is set
# to (current_mean - 0.5), weight=1.0. Then loss_coefficients gives exactly
# coeff = 2*1*(mean - target) = 1.0, so the seed fed into the backward pass
# is EXACTLY ∂mean/∂field (the area-weighted-mean adjoint), and the returned
# "gradient" is the RAW d(flux_mean)/d(theta) -- not scaled by any residual.
# This is what lets a single number be compared directly against literature
# sensitivities like d(olr)/d(tau0_equator)=-2.57 without needing to know a
# target/weight convention.
# ─────────────────────────────────────────────────────────────────────────

function raw_sensitivity_cfg(flux_key::Symbol, means::Dict{Symbol,Float32})
    LossConfig([flux_key];
        targets = Dict(flux_key => means[flux_key] - 0.5f0),
        weights = Dict(flux_key => 1f0))
end

# Need current means first to build the config (target is state-dependent but
# irrelevant to the resulting raw gradient as long as coeff ends up = 1).
means0, _ = SpeedyCalibration.compute_flux_means(sim.variables, [:olr, :lrd])
compute_gradients! = SpeedyCalibration.compute_gradients!
@printf("\nSpun-up state: olr=%.3f W/m^2, lrd=%.3f W/m^2\n", means0[:olr], means0[:lrd])

olr_cfg = raw_sensitivity_cfg(:olr, means0)
lrd_cfg = raw_sensitivity_cfg(:lrd, means0)

# ─────────────────────────────────────────────────────────────────────────
# 3) Single-timestep gradients (compute_gradients!) -- sanity check against
#    the documented single-parameter-sweep numbers:
#      d(olr)/d(tau0_equator) = -2.57,  d(lrd)/d(tau0_equator) = +11.18
#      d(olr)/d(fl)           = -15.1,  d(lrd)/d(fl)           = +125.9
# ─────────────────────────────────────────────────────────────────────────

println("\n=== Single-timestep gradients (compute_gradients!) ===")
grads_olr_1step, _, _ = compute_gradients!(sim.variables, sim.model, olr_cfg, param_specs)
grads_lrd_1step, _, _ = compute_gradients!(sim.variables, sim.model, lrd_cfg, param_specs)

for (i, spec) in enumerate(param_specs)
    @printf("%-14s  d(olr)/dθ = %10.4f   d(lrd)/dθ = %10.4f   ratio(lrd/olr) = %8.4f\n",
            spec.name, grads_olr_1step[i], grads_lrd_1step[i],
            grads_lrd_1step[i] / grads_olr_1step[i])
end

# ─────────────────────────────────────────────────────────────────────────
# 4) N=20 checkpointed multi-step gradients (compute_gradients_checkpointed!)
#    at the SAME spun-up state (compute_gradients_checkpointed! internally
#    saves/restores full state including the clock, so sim is unaffected by
#    either call, and both calls below start from the identical state).
# ─────────────────────────────────────────────────────────────────────────

println("\n=== N=$N checkpointed gradients (compute_gradients_checkpointed!) ===")
grads_olr_N, means_olr_N, loss_olr_N = compute_gradients_checkpointed!(
    sim.variables, sim.model, olr_cfg, param_specs, N)
grads_lrd_N, means_lrd_N, loss_lrd_N = compute_gradients_checkpointed!(
    sim.variables, sim.model, lrd_cfg, param_specs, N)

println("(N-steps-ahead means used to build seeds: olr=$(means_olr_N[:olr]), lrd=$(means_lrd_N[:lrd]))")
any(!isfinite, grads_olr_N) && println("WARNING: non-finite values in grads_olr_N: $grads_olr_N")
any(!isfinite, grads_lrd_N) && println("WARNING: non-finite values in grads_lrd_N: $grads_lrd_N")

for (i, spec) in enumerate(param_specs)
    @printf("%-14s  d(olr)/dθ = %10.4f   d(lrd)/dθ = %10.4f   ratio(lrd/olr) = %8.4f\n",
            spec.name, grads_olr_N[i], grads_lrd_N[i],
            grads_lrd_N[i] / grads_olr_N[i])
end

# ─────────────────────────────────────────────────────────────────────────
# 5) Comparison table: ratio at N=1 vs ratio at N=20, and the shift.
# ─────────────────────────────────────────────────────────────────────────

println("\n=== Ratio comparison: [d(lrd)/dθ]/[d(olr)/dθ] ===")
@printf("%-14s  %12s  %12s  %10s\n", "param", "ratio(N=1)", "ratio(N=20)", "Δratio %")
results = NamedTuple[]
for (i, spec) in enumerate(param_specs)
    r1 = grads_lrd_1step[i] / grads_olr_1step[i]
    rN = grads_lrd_N[i] / grads_olr_N[i]
    pct = 100 * (rN - r1) / abs(r1)
    @printf("%-14s  %12.4f  %12.4f  %9.1f%%\n", spec.name, r1, rN, pct)
    push!(results, (
        name = spec.name,
        olr_1step = grads_olr_1step[i], lrd_1step = grads_lrd_1step[i], ratio_1step = r1,
        olr_N = grads_olr_N[i], lrd_N = grads_lrd_N[i], ratio_N = rN,
        ratio_pct_change = pct,
    ))
end

save_path = joinpath(@__DIR__, "output", "lrd_olr_coupling_probe.jld2")
mkpath(dirname(save_path))
jldsave(save_path;
    N, dt_ms = Millisecond(clock.Δt).value, spinup_days,
    means0_olr = means0[:olr], means0_lrd = means0[:lrd],
    param_names = [String(s.name) for s in param_specs],
    grads_olr_1step, grads_lrd_1step, grads_olr_N, grads_lrd_N,
    results,
)
println("\nSaved results to $save_path")
