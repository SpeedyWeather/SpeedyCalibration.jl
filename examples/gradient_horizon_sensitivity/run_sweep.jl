# Gradient-horizon sensitivity sweep.
#
# Question: as the checkpointed-AD gradient horizon N (number of `time_step!`
# calls differentiated through in a single `Enzyme.autodiff` reverse pass, via
# `compute_gradients_checkpointed!`) grows from N=1 (matching the single-step
# `compute_gradients!` used throughout this investigation's successful runs)
# up to wherever it stops being numerically valid, how does the AD-computed
# sensitivity d(flux)/d(theta) behave -- for a representative mix of
# shortwave AND longwave parameters, evaluated at multiple independent points
# in the model's trajectory, not just one snapshot?
#
# This generalizes `examples/checkpointed_multistep_gradients/lrd_olr_coupling_probe.jl`
# (3 longwave params, N=1 vs N=20 only, ONE snapshot) along three axes:
#   1. 6 params spanning both SW and LW processes, not just 3 LW ones.
#   2. A full N grid (not just two points), pushed until it demonstrably breaks.
#   3. >=5 independent sample states spread across continued integration, so
#      findings are means+spread across states, not one lucky/unlucky instant.
#
# See the notebook `gradient_horizon_sensitivity.ipynb` in this directory for
# the full write-up; this script only computes and saves the raw results.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using SpeedyCalibration, SpeedyWeather, Dates, Printf, JLD2, Statistics

const OUTDIR = joinpath(@__DIR__, "output")
mkpath(OUTDIR)
const RESULTS_PATH = joinpath(OUTDIR, "gradient_horizon_sweep_results.jld2")
const LOG_PATH = joinpath(OUTDIR, "run_sweep.log")

function tprint(io, msg)
    line = "[$(now())] $msg"
    println(line)
    println(io, line)
    flush(io)
    flush(stdout)
end

logio = open(LOG_PATH, "a")

# ─────────────────────────────────────────────────────────────────────────
# 1) Model setup: standard T31/L8 PrimitiveWetModel, daily_cycle=true,
#    seasonal_cycle=false -- the config used throughout this investigation's
#    stable runs (thesis_15param_shortwave.ipynb etc.), NOT tied to any one
#    calibration script's hyperparameters.
# ─────────────────────────────────────────────────────────────────────────

sg     = SpectralGrid(trunc=31, nlayers=8)
planet = Earth(sg; daily_cycle=true, seasonal_cycle=false)
model  = PrimitiveWetModel(sg; planet=planet)
sim    = initialize!(model)
sim.variables.prognostic.clock.time = DateTime(2000, 3, 21)
SpeedyWeather.initialize!(sim; period=Day(365*100), output=false)

clock = sim.variables.prognostic.clock
steps_per_day = ceil(Int, Millisecond(Day(1)).value / Millisecond(clock.Δt).value)
tprint(logio, "Δt = $(clock.Δt), steps_per_day = $steps_per_day")

function step_days!(sim, days, steps_per_day)
    for _ in 1:(days * steps_per_day)
        SpeedyWeather.time_step!(sim)
    end
end

# ─────────────────────────────────────────────────────────────────────────
# 2) Spin up once, then take >=5 independent sample states spread across
#    continued integration (every 15 days), each saved via the package's own
#    `_save_full_state` helper (same one `compute_gradients_checkpointed!`
#    uses internally) so each is a faithful, restorable snapshot.
# ─────────────────────────────────────────────────────────────────────────

const SPINUP_DAYS     = 30
const SAMPLE_GAP_DAYS = 15
const N_SAMPLES        = 5

tprint(logio, "Spinning up $SPINUP_DAYS days...")
t0 = time()
step_days!(sim, SPINUP_DAYS, steps_per_day)
tprint(logio, @sprintf("Spinup complete in %.1f s.", time() - t0))

sample_states = NamedTuple[]
for i in 1:N_SAMPLES
    if i > 1
        tprint(logio, "Advancing $SAMPLE_GAP_DAYS more days to sample $i/$N_SAMPLES...")
        t0 = time()
        step_days!(sim, SAMPLE_GAP_DAYS, steps_per_day)
        tprint(logio, @sprintf("  done in %.1f s.", time() - t0))
    end
    day = SPINUP_DAYS + (i - 1) * SAMPLE_GAP_DAYS
    means_i, _ = SpeedyCalibration.compute_flux_means(sim.variables, [:olr, :lrd])
    tprint(logio, "Sample $i at day $day: olr=$(means_i[:olr]), lrd=$(means_i[:lrd])")
    push!(sample_states, (
        day   = day,
        state = SpeedyCalibration._save_full_state(sim.variables),
        olr0  = means_i[:olr],
        lrd0  = means_i[:lrd],
    ))
end

# ─────────────────────────────────────────────────────────────────────────
# 3) Parameters: 3 shortwave + 3 longwave, paths/bounds copied verbatim from
#    examples/trenberth_full.jl (SW) and the coupling probe (LW).
# ─────────────────────────────────────────────────────────────────────────

param_specs = [
    ParamSpec(:cloud_albedo, [:shortwave_radiation, :clouds, :cloud_albedo];
        bounds=(0.25f0, 0.95f0), initial=0.60f0),
    ParamSpec(:absorptivity_water_vapor, [:shortwave_radiation, :transmissivity, :absorptivity_water_vapor];
        bounds=(60f0, 140f0), initial=75f0, grad_scale=0.01f0),
    ParamSpec(:ozone_absorption, [:shortwave_radiation, :radiative_transfer, :ozone_absorption];
        bounds=(0.002f0, 0.020f0), initial=0.01f0),
    ParamSpec(:tau0_equator, [:longwave_radiation, :transmissivity, :τ₀_equator];
        bounds=(2f0, 12f0), initial=6f0),
    ParamSpec(:fl, [:longwave_radiation, :transmissivity, :fₗ];
        bounds=(0.0f0, 0.5f0), initial=0.1f0),
    ParamSpec(:emissivity_ocean, [:longwave_radiation, :radiative_transfer, :emissivity_ocean];
        bounds=(0.80f0, 1.00f0), initial=0.98f0),
]
const PARAM_KIND = Dict(
    :cloud_albedo => :SW, :absorptivity_water_vapor => :SW, :ozone_absorption => :SW,
    :tau0_equator => :LW, :fl => :LW, :emissivity_ocean => :LW,
)

# ─────────────────────────────────────────────────────────────────────────
# Raw-sensitivity trick (verbatim from lrd_olr_coupling_probe.jl): a
# single-flux LossConfig with target = current_mean - 0.5, weight = 1
# gives seed coefficient coeff = 2*(mean_at_seed_time - target).
#
# IMPORTANT CORRECTNESS FIX vs. the original probe script: the probe only
# ever compared N=1 vs N=20 at ONE state, where the flux mean barely drifts
# between "now" and "N steps ahead", so coeff stayed close enough to 1.0 to
# treat the returned gradient as the raw sensitivity directly. Here N ranges
# up to 250 steps (days of model time), over which the flux mean can drift by
# more than the assumed 0.5 W/m^2, so coeff is NOT reliably ~1.0. We correct
# for this exactly: `compute_gradients_checkpointed!` returns `means`, the
# ACTUAL N-steps-ahead flux means it built the seed from, so the true
# realized coefficient can be reconstructed after the fact from `cfg` + that
# returned `means`, and the raw sensitivity recovered by dividing it out --
# no approximation, exact for every N.
# ─────────────────────────────────────────────────────────────────────────

function raw_sensitivity_cfg(flux_key::Symbol, means::Dict{Symbol,Float32})
    LossConfig([flux_key]; targets=Dict(flux_key => means[flux_key] - 0.5f0), weights=Dict(flux_key => 1f0))
end

function realized_coeff(cfg::LossConfig, flux_key::Symbol, means_N::Dict{Symbol,Float32})
    2f0 * cfg.weights[flux_key] * (means_N[flux_key] - cfg.targets[flux_key])
end

# ─────────────────────────────────────────────────────────────────────────
# 4) N sweep. Garbage = finite but |value| > GARBAGE_ABS_THRESHOLD (raw
#    sensitivities in this problem are O(0.1-100) by literature/N=1 values;
#    anything past 1e6 is unambiguous numerical blow-up, not signal).
# ─────────────────────────────────────────────────────────────────────────

const N_GRID = [1, 3, 5, 10, 20, 40, 80, 150, 250]
const GARBAGE_ABS_THRESHOLD = 1f6
const FLUX_KEYS = (:olr, :lrd)

# rows: one per (N, state, flux_key, param) combination
rows = NamedTuple[]

function status_of(x::Float32)
    isnan(x) && return :nan
    !isfinite(x) && return :inf
    abs(x) > GARBAGE_ABS_THRESHOLD && return :garbage
    return :ok
end

tprint(logio, "Starting N sweep: $N_GRID")
compute_gradients_checkpointed! = SpeedyCalibration.compute_gradients_checkpointed!

for N in N_GRID
    global rows
    tprint(logio, "=== N=$N ===")
    n_ok = 0
    n_total = 0
    for (si, samp) in enumerate(sample_states)
        for flux_key in FLUX_KEYS
            SpeedyCalibration._restore_full_state!(sim.variables, samp.state)
            means_now, _ = SpeedyCalibration.compute_flux_means(sim.variables, [flux_key])
            cfg = raw_sensitivity_cfg(flux_key, means_now)

            t0 = time()
            local grads, means_N, loss
            crashed = false
            errmsg = ""
            try
                grads, means_N, loss = compute_gradients_checkpointed!(
                    sim.variables, sim.model, cfg, param_specs, N)
            catch e
                crashed = true
                errmsg = sprint(showerror, e)
                grads = fill(NaN32, length(param_specs))
                means_N = Dict{Symbol,Float32}(flux_key => NaN32)
            end
            elapsed = time() - t0

            if crashed
                tprint(logio, "  sample=$si ($(samp.day)d) flux=$flux_key N=$N CRASHED: $errmsg")
                for spec in param_specs
                    n_total += 1
                    push!(rows, (N=N, sample=si, day=samp.day, flux=flux_key,
                        param=spec.name, kind=PARAM_KIND[spec.name],
                        raw_grad=NaN32, status=:crashed, elapsed=elapsed, coeff=NaN32))
                end
                continue
            end

            coeff = realized_coeff(cfg, flux_key, means_N)
            # coeff is NOT expected to equal 1.0 for N>1: it is the REALIZED seed
            # coefficient 2*w*(means_N[flux]-cfg.targets[flux]), evaluated at the
            # N-steps-ahead state, whereas cfg.targets was built from the "now" state's
            # mean -- these only coincide when the flux hasn't moved between "now" and
            # "N steps ahead" (true for N=1, not guaranteed for larger N). We don't rely
            # on coeff~=1; raw_grad below divides it out exactly, which is what makes the
            # extracted sensitivity correct regardless of how far coeff drifts. Saved
            # per-row (not just logged as text) so this is auditable directly from the
            # JLD2 without re-deriving from the log.
            for (i, spec) in enumerate(param_specs)
                n_total += 1
                raw_grad = isfinite(coeff) && abs(coeff) > 1f-8 ? grads[i] / coeff : NaN32
                st = status_of(Float32(raw_grad))
                st == :ok && (n_ok += 1)
                push!(rows, (N=N, sample=si, day=samp.day, flux=flux_key,
                    param=spec.name, kind=PARAM_KIND[spec.name],
                    raw_grad=Float32(raw_grad), status=st, elapsed=elapsed, coeff=Float32(coeff)))
            end
            tprint(logio, @sprintf("  sample=%d (%dd) flux=%s N=%-4d coeff=%.4f elapsed=%.1fs",
                si, samp.day, flux_key, N, coeff, elapsed))
        end
    end
    tprint(logio, @sprintf("N=%d: %d/%d (param,sample,flux) combos valid (%.0f%%)",
        N, n_ok, n_total, 100 * n_ok / max(n_total, 1)))

    # Save incrementally after every N so a partial run is still usable.
    jldsave(RESULTS_PATH;
        rows, N_GRID_completed = [r.N for r in rows] |> unique,
        param_names = [String(s.name) for s in param_specs],
        param_kind = PARAM_KIND,
        sample_days = [s.day for s in sample_states],
        dt_ms = Millisecond(clock.Δt).value,
        spinup_days = SPINUP_DAYS, sample_gap_days = SAMPLE_GAP_DAYS,
        garbage_abs_threshold = GARBAGE_ABS_THRESHOLD,
    )
    tprint(logio, "Saved results (N up to $N) to $RESULTS_PATH")

    if n_ok == 0
        tprint(logio, "N=$N: 0% valid across all (param,sample,flux) combos -- treating as the empirical ceiling, stopping sweep.")
        break
    end
end

tprint(logio, "Sweep complete.")
close(logio)
