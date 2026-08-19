"""
    checkpointed_timesteps!(variables, model, N_steps, checkpoint_scheme)

Advance `variables` by `N_steps` timesteps, wrapped in `Checkpointing.jl`'s
`@ad_checkpoint` macro so the whole loop can be differentiated in a single
`Enzyme.autodiff` reverse-mode pass (via `Checkpointing.Revolve`) instead of
storing every intermediate state.

Mirrors `SpeedyWeather.jl`'s own `test/differentiability/sensitivity_examples/checkpointed_sensitivity.jl`
(`SpeedyWeather.jl` PR #1191, Max Gelbrecht): each iteration does the dynamics/physics
step and the clock step as two separate `SpeedyWeather.time_step!` calls, both inside
the checkpointed loop.
"""
function checkpointed_timesteps!(variables, model, N_steps, checkpoint_scheme::Checkpointing.Scheme)
    @ad_checkpoint checkpoint_scheme for _ in 1:N_steps
        SpeedyWeather.time_step!(variables, model.time_stepping, model)
        SpeedyWeather.time_step!(variables.prognostic.clock, model.time_stepping)
    end
    return nothing
end

# Save/restore helpers shared with `compute_gradients!` semantics: a call must leave
# `variables`/`model` exactly as it found them, clock included (the checkpointed loop
# advances the clock as part of what's differentiated, unlike the single-timestep path).

function _save_full_state(variables)
    progn = variables.prognostic
    return (
        vor   = deepcopy(progn.vorticity),
        div   = deepcopy(progn.divergence),
        temp  = deepcopy(progn.temperature),
        hum   = deepcopy(progn.humidity),
        pres  = deepcopy(progn.pressure),
        ocean = deepcopy(progn.ocean),
        land  = deepcopy(progn.land),
        clock = deepcopy(progn.clock),
        grid  = deepcopy(variables.grid),
    )
end

function _restore_full_state!(variables, saved)
    progn = variables.prognostic
    progn.vorticity   .= saved.vor
    progn.divergence  .= saved.div
    progn.temperature .= saved.temp
    progn.humidity    .= saved.hum
    progn.pressure    .= saved.pres
    for key in keys(progn.ocean); progn.ocean[key] .= saved.ocean[key]; end
    for key in keys(progn.land);  progn.land[key]  .= saved.land[key];  end
    # Clock is a mutable struct with several counters/timestamps (time, start,
    # rotation_time, orbit_time, step_counter, time_step_counter, n_steps, ...) —
    # copy every field generically rather than hand-picking a subset that could
    # silently miss one and leave the clock in a state the simulation never
    # actually reached.
    for fn in fieldnames(typeof(progn.clock))
        setproperty!(progn.clock, fn, getproperty(saved.clock, fn))
    end
    for fn in fieldnames(typeof(variables.grid))
        dst = getfield(variables.grid, fn)
        src = getfield(saved.grid, fn)
        dst isa AbstractArray && copyto!(dst, src)
    end
    return nothing
end

"""
    compute_gradients_checkpointed!(variables, model, loss_config, param_specs, N)
        → (gradients, means, loss)

Checkpointed **N-step** reverse-mode AD pass (Enzyme + `Checkpointing.jl`'s `Revolve(N)`
scheme), as opposed to `compute_gradients!`'s single timestep. Returns the same shape of
result — `∂L/∂θ` per `ParamSpec`, the area-weighted flux means, and the scalar loss — but
the loss and its gradient are evaluated on the state **N steps ahead** of the simulation's
current position, and differentiated all the way back through all N steps in one
`Enzyme.autodiff` call rather than being approximated by averaging single-step gradients.

# Mechanics

Unlike `compute_gradients!`, which reuses the diagnostics already sitting in `variables`
from the *previous* step, this function has no such state available for a point N steps in
the future — it doesn't exist yet. So it:

1. Saves the full state (progn fields, grid, ocean/land, **and clock** — the checkpointed
   loop advances the clock as part of what gets differentiated).
2. Runs `N` steps forward **undifferentiated** to reach the state the loss will be
   evaluated on, and reads off the flux means/coefficients there.
3. Restores the saved state exactly (undoing that throwaway run).
4. Seeds `dvariables.parameterizations` with those coefficients (same per-ring
   area-weighted seeding as `compute_gradients!`) and runs the *actual* checkpointed
   `Enzyme.autodiff` pass over `checkpointed_timesteps!`, starting again from the restored
   state.
5. Restores the saved state again, so the call has no net effect on `variables`/`model`.

This means one extra undifferentiated N-step forward pass compared to a hypothetical
version that could seed without knowing the future state first — real, but small next to
Revolve's own internal recomputation.

# The Enzyme Attributor segfault (Julia < 1.12)

Enzyme runs the LLVM Attributor optimization pass by default on Julia < 1.12. Stepping the
clock inside the `@ad_checkpoint` loop sends the Attributor's `AAPotentialValues` analysis
into unbounded recursion (`AAPotentialValuesFloating::updateImpl` → `getAssumedSimplified`
→ ... → itself), overflowing the C++ stack — this surfaces as a segfault with no Julia-level
error, not a clean crash. `Enzyme.Compiler.RunAttributor[]` is toggled off for the duration
of the `autodiff` call (and restored afterward, since it's a process-global flag that would
otherwise also affect `compute_gradients!`'s compilation if called later in the same
session). Root cause identified and fixed upstream by Max Gelbrecht,
`SpeedyWeather.jl` PR #1191.

# ⚠ Gradient blow-up: NOT free to push N up

Empirically (T33/L8 `PrimitiveWetModel`, Δt=2400s, one-hot sensitivity seeds, this exact
`checkpointed_timesteps!` loop): the run itself completes without crashing at any `N` tried,
**but the gradient silently blows up to all-NaN well before that** —

| N   | gradient max \\|value\\| | NaN?          |
|-----|--------------------------|---------------|
| 3   | 0.0058                   | none          |
| 20  | 0.26 (already ~45× the N=3 value for a 6.7× step increase — faster than linear) | none |
| 200 | —                        | **100% NaN**  |
| 360 | —                        | **100% NaN**  |

This is the classic adjoint-instability problem for chaotic nonlinear systems: reverse-mode
sensitivities amplify along the model's positive Lyapunov exponents, compounded here by
moist convection's threshold/switch behaviour (rain onset, condensation thresholds — the
same kind of non-smoothness already known to zero out `conv_time_scale`/`lsc_rh_threshold`
gradients at the single-step level). **This is precisely the failure mode this package's
own module docstring says it was designed to avoid** ("averaging single-timestep
Enzyme.jl gradients, instead of differentiating through long, chaotically unstable
trajectories") — this integration doesn't remove that constraint, it just makes it
possible to hit deliberately and measure where it bites for a specific model config.
**A usable `N` for this model config is therefore well under 200; the actual ceiling
(between ~20 and ~200) has not yet been bisected.** Do not assume a larger N is safe just
because a run completes.
"""
function compute_gradients_checkpointed!(
        variables,
        model,
        cfg::LossConfig,
        param_specs::Vector{ParamSpec},
        N::Integer,
    )
    checkpoint_scheme = Revolve(N)
    saved = _save_full_state(variables)

    # 1) Undifferentiated N-step forward pass, purely to know what state the loss
    #    will be evaluated on (needed to build the seed before the real AD call).
    model.feedback.nans_detected = false
    for _ in 1:N
        SpeedyWeather.time_step!(variables, model.time_stepping, model)
        SpeedyWeather.time_step!(variables.prognostic.clock, model.time_stepping)
    end

    means, wsums = compute_flux_means(variables, cfg.flux_keys)
    if any(!isfinite(means[k]) for k in cfg.flux_keys)
        _restore_full_state!(variables, saved)
        nan_means = Dict{Symbol,Float32}(k => NaN32 for k in cfg.flux_keys)
        return fill(0f0, length(param_specs)), nan_means, Inf32
    end

    loss   = compute_loss(means, cfg)
    coeffs = loss_coefficients(means, cfg)

    # 2) Undo the throwaway pass; the real (differentiated) trajectory starts fresh
    #    from the true current state.
    _restore_full_state!(variables, saved)
    model.feedback.nans_detected = false

    dvariables = Enzyme.make_zero(variables)
    dmodel     = Enzyme.make_zero(model)

    # Seed cotangents per ring, on the (not-yet-recomputed) parameterizations field —
    # identical convention to `compute_gradients!`, just fed by the N-steps-ahead means.
    param = variables.parameterizations
    first_field = getproperty(param, FLUX_FIELD[cfg.flux_keys[1]])
    _grid  = first_field.grid
    _gw    = Float32.(RingGrids.gaussian_weights(_grid.nlat_half))
    _rings = eachring(_grid)

    dparam = dvariables.parameterizations
    for (j, ring) in enumerate(_rings)
        nlons_j = length(ring)
        seeds = Dict{Symbol,Float32}(
            k => coeffs[k] * _gw[j] / (Float32(nlons_j) * wsums[k]) for k in cfg.flux_keys
        )
        for i in ring, k in cfg.flux_keys
            field  = getproperty(param,  FLUX_FIELD[k])
            dfield = getproperty(dparam, FLUX_FIELD[k])
            dfield[i] = isfinite(field[i]) ? seeds[k] : 0f0
        end
    end

    # 3) The real checkpointed reverse pass, guarded against the Attributor segfault.
    prev_run_attributor = Enzyme.Compiler.RunAttributor[]
    Enzyme.Compiler.RunAttributor[] = false
    try
        Enzyme.autodiff(Enzyme.Reverse, checkpointed_timesteps!, Const,
            Duplicated(variables, dvariables),
            Duplicated(model, dmodel),
            Const(N), Const(checkpoint_scheme))
    finally
        Enzyme.Compiler.RunAttributor[] = prev_run_attributor
    end

    model.feedback.nans_detected = false
    _restore_full_state!(variables, saved)

    gradients = Float32[Float32(get_by_path(dmodel, spec.path)) for spec in param_specs]
    return gradients, means, loss
end
