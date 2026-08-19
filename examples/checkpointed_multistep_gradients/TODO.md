# Checkpointed multi-step gradients

Status: **integrated and numerically validated** (`src/gradients_checkpointed.jl`,
`compute_gradients_checkpointed!`). Not yet wired into `calibrate!`'s training loop, and
the gradient-blow-up caveat below means it isn't clearly usable for the `lrd`/`srd`
investigation as-is — read that section before building on this.

## The idea

`compute_gradients!` (`src/gradients.jl`) differentiates through exactly **one**
`time_step!` call — the docstring says so directly ("Single-timestep reverse-mode AD
pass"). Everything `batch_days`/`samples_per_batch` currently do
(`examples/trenberth_batchdays_gradcount_sweep/`) only changes how much *forward-only,
undifferentiated* time elapses between single-step gradient snapshots, and how many of
those snapshots get averaged. The gradient itself never spans more than one timestep.

`Checkpointing.jl`'s `@ad_checkpoint` macro with a `Revolve(N)` scheme wraps a loop of `N`
consecutive `time_step!` calls, all differentiated in a single `Enzyme.autodiff` call.
Revolve is a classic adjoint-checkpointing algorithm (Griewank & Walther) — trades some
recomputation for memory so that backpropagating through many timesteps stays tractable
instead of needing the full forward trajectory in memory. This is now
`compute_gradients_checkpointed!` in `src/gradients_checkpointed.jl`.

Source: `test/differentiability/sensitivity_examples/checkpointed_sensitivity.jl` in
SpeedyWeather.jl, originally PR #876 (Max Gelbrecht), fixed up in PR #1191 (also Max) —
this second PR found and fixed a real segfault root cause, not just an API rename, see
below.

## Why this matters for this project

This whole investigation's `batch_days`/`samples_per_batch` work has been an indirect
proxy for a question a real multi-step gradient could answer directly: does giving the
gradient itself more elapsed time change how it weighs fast fluxes (`osr`, `olr`) against
slow ones (`srd`, `lrd`)? A checkpointed multi-step gradient lets training see the actual
multi-timestep sensitivity directly. **However**, see the gradient-blow-up finding below —
the direct evidence so far argues *against* "just extend N" as a fix, not for it.

## What's done

- `Project.toml`: `SpeedyWeather` compat bumped `"0.21"` → `"0.22"` (registered release,
  no dev/git pin needed — see below), `Checkpointing = "0.11, 0.12"` added as a new dep.
- `src/gradients_checkpointed.jl`: `compute_gradients_checkpointed!(variables, model,
  loss_config, param_specs, N)`, mirroring `compute_gradients!`'s signature/return shape
  (`gradients, means, loss`), using the package's real flux-mean loss and per-ring
  area-weighted seeding — **not** the one-hot toy seed from the upstream example.
- Fixed a **pre-existing break**, not something this integration introduced:
  `SpeedyWeather.timestep!` was removed between 0.21 and 0.22, replaced by the
  `time_step!` family. This broke `training.jl`'s spinup/training-loop stepping (3 call
  sites) and `compute_gradients!`'s own `Enzyme.autodiff` call — i.e. the *existing*
  single-step gradient path was also broken by the version bump, independent of anything
  checkpointing-related. Both fixed:
  - `training.jl`: `SpeedyWeather.timestep!(sim)` → `SpeedyWeather.time_step!(sim)`
    (convenience wrapper, does dynamics+physics step, clock step, feedback, output,
    callbacks — a clean drop-in).
  - `gradients.jl`: `Enzyme.autodiff(Reverse, SpeedyWeather.timestep!, ...)` →
    `Enzyme.autodiff(Reverse, SpeedyWeather.time_step!, Const, Duplicated(variables,
    dvariables), Const(model.time_stepping), Duplicated(model, dmodel))` — this variant
    of `time_step!` does dynamics+physics only, no clock step, matching what the old
    `timestep!(variables, dt, model)` actually did (clock was never part of what
    `compute_gradients!` differentiates or restores).
- Validated end-to-end (T32/L8 `PrimitiveWetModel`, `Earth(daily_cycle=true)`, 5-day
  spinup, `OSR_LOSS`, N=5, 2 params):
  - `loss` matches the closed-form check exactly (mean OSR 85.74 vs target 101.9 →
    residual² ≈ 261.2, matches the returned `261.18246`).
  - Gradients finite, no NaN.
  - Called twice in a row on the same live `sim`: second call's gradients matched the
    first (`≈`, small float-level differences from non-associative reductions, not
    drift) — confirms `_save_full_state`/`_restore_full_state!` (progn fields, grid,
    ocean/land, **and clock** — the checkpointed loop advances the clock as part of what
    gets differentiated, unlike `compute_gradients!`) leave the simulation exactly where
    they found it, so repeated calls in a training loop won't corrupt state.

## Resolved: the segfault was a real, identified bug — not flaky research code

The original caveat here was the upstream comment "occasionally this gives a
SegmentationFault (especially on x86 and for large N), but not always." Root cause, found
by Max Gelbrecht (SpeedyWeather.jl PR #1191): Enzyme runs the LLVM Attributor optimization
pass by default on Julia < 1.12. Stepping the clock inside the `@ad_checkpoint` loop sends
the Attributor's `AAPotentialValues` analysis into unbounded recursion
(`AAPotentialValuesFloating::updateImpl` → `getAssumedSimplified` → ... → itself),
overflowing the C++ stack — a segfault with no Julia-level error, hence "occasional" and
unexplained-looking. Fix: `Enzyme.Compiler.RunAttributor[] = false` around the `autodiff`
call. `compute_gradients_checkpointed!` does this, scoped with a `try`/`finally` that
restores the previous value afterward (it's a process-global flag — leaving it off would
silently also affect `compute_gradients!`'s compilation if called later in the same
session).

## No dev/git pin needed — 0.22 is a normal registered release

Checked directly: PR #1191 (the segfault fix + API-syntax update) touches **only** the
example scripts and their local `Manifest.toml` — zero changes to SpeedyWeather's `src/`.
The Attributor toggle is a pure Enzyme.jl runtime setting; the `Variables`-unified
`time_step!` API it's built on is already in the registered `v0.22.0`. So this package can
depend on a normal tagged SpeedyWeather release, not an unreleased `main`/commit pin.

## ⚠ Gradient blow-up: the open, load-bearing risk

Before integrating, N was swept with the raw upstream one-hot script (T33/L8
`PrimitiveWetModel`, Δt=2400s) to check for the segfault. It didn't segfault at any N
tried — but the **gradient itself silently blew up to all-NaN** well before that:

| N   | gradient max \|value\| | NaN?          |
|-----|--------------------------|---------------|
| 3   | 0.0058                   | none          |
| 20  | 0.26 (≈45× the N=3 value for a 6.7× step increase — faster than linear) | none |
| 200 | —                        | **100% NaN**  |
| 360 (≈ Max's reported working 20-day trajectory) | — | **100% NaN** |

Not yet re-run through `compute_gradients_checkpointed!` itself at N>5 (each attempt costs
real wall-clock — see below), but there's no reason to expect the real flux-loss seed
would behave differently in kind, only possibly in the exact N where it happens, since the
instability is a property of the model's forward dynamics under reverse-mode
differentiation, not of which cotangent seed is used.

This is the classic adjoint-instability problem for chaotic nonlinear systems:
reverse-mode sensitivities amplify along the model's positive Lyapunov exponents,
compounded by moist convection's threshold/switch behaviour (rain onset, condensation
thresholds — the same non-smoothness already known to zero out `conv_time_scale`/
`lsc_rh_threshold` gradients at the single-step level, see main `MEMORY.md`).
**This is exactly the failure mode this package's own module docstring says it exists to
avoid**: "averaging single-timestep Enzyme.jl gradients, instead of differentiating
through long, chaotically unstable trajectories." Integrating checkpointing doesn't
remove that constraint — it makes it possible to hit deliberately and measure.

**Before using this for the `lrd`/`srd` investigation: bisect where the real (flux-loss,
not one-hot) gradient actually goes bad, on the calibration-relevant param set,** and
treat "does a longer-N gradient survive at all" as a prerequisite finding, not an assumed
yes. If the real ceiling is also well under N≈200 (~a few hours of model time at this
Δt), a 20-day-scale multi-step gradient — the thing that would most directly test the
`lrd`/`srd` hypothesis — may simply not be reachable without first taming the underlying
instability (gradient clipping mid-trajectory? a smoothed convection scheme? something
else) — which would be a materially different, harder project than "wire up
checkpointing."

## Cost: compile time depends heavily on what's being differentiated

Two very different numbers were measured and both are real, for different things:

- Raw upstream one-hot script, bare `PrimitiveWetModel` (T33/L8, no `Earth`/calibration
  wrapper): ~2,300–2,460s (~40 min) to compile, **independent of N** (N=3, 20, 200, 360 all
  compiled in roughly the same time) — only the non-compiled remainder scaled with N
  (~1s → ~90s over that N range).
- `compute_gradients_checkpointed!` through this package's actual model construction
  (`Earth(daily_cycle=true)`, the two-forward-pass structure needed to seed from a
  not-yet-computed future state): **10,751s (≈3 hours) at N=5**. Not yet checked whether
  this also stays flat across N, or whether the extra structure (two `time_step!` calls
  per differentiated iteration instead of one bundled `timestep!`, the full parameterization
  stack in `model`) makes compile cost N-dependent too — that's now the more urgent
  unknown for planning any real experiment, more urgent than the earlier "does it scale
  with N" question, since 3 hours per distinct N is a very different planning constraint
  than 40 minutes.
- Once compiled, a second call with the same N is cheap regardless: 1.4s here, 1.5s in the
  raw script — consistent across both, so the cost is a one-time-per-process tax, just a
  much bigger one for the real integration than the toy script suggested.

## Before running any real N-sweep experiment

1. Bisect the real (flux-loss-seeded) gradient's NaN ceiling — this determines whether
   the whole approach is viable for the `lrd`/`srd` question at all before spending more
   compile-time budget on it.
2. Check whether `compute_gradients_checkpointed!`'s ~3-hour compile cost is flat across N
   (as the raw script's was) or grows — this determines whether an N-sweep is even
   practically schedulable.
3. Only then consider wiring into `calibrate!`'s actual training loop (currently this is a
   standalone function, callable but not part of the batch/optimizer loop) — that's a
   further step, since the loop's "step forward, then get a gradient of the current
   state" structure doesn't map directly onto "step forward N, get a gradient of the state
   N steps from wherever training currently is."
