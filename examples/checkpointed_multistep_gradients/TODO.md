# TODO: checkpointed multi-step gradients

Not started. No code here yet, this is a placeholder for a real future direction.

## The idea

`compute_gradients!` (`src/gradients.jl`) differentiates through exactly **one**
`timestep!` call — the docstring says so directly ("Single-timestep reverse-mode
AD pass"). Everything `batch_days`/`samples_per_batch` currently do
(`examples/trenberth_batchdays_gradcount_sweep/`) only changes how much
*forward-only, undifferentiated* time elapses between single-step gradient
snapshots, and how many of those snapshots get averaged. The gradient itself
never spans more than one timestep.

There's a working (if experimental) example upstream in SpeedyWeather.jl of a
genuinely different approach: `Checkpointing.jl`'s `@ad_checkpoint` macro with a
`Revolve(N)` scheme, wrapping a loop of `N` consecutive `timestep!` calls, all
differentiated in a single `Enzyme.autodiff` call. Revolve is a classic
adjoint-checkpointing algorithm (Griewank & Walther) — trades some
recomputation for memory so that backpropagating through many timesteps stays
tractable instead of needing the full forward trajectory in memory.

Source: `test/differentiability/sensitivity_examples/checkpointed_sensitivity.jl`
in the SpeedyWeather.jl PR #876 branch ("Sensitivity analysis examples",
Max Gelbrecht). Not part of the SpeedyWeather version this package currently
depends on (`compat = "0.21"`) — found via a local checkout at
`SpeedyWeather-pr876/`, not from a released version.

## Why this matters for this project

This whole investigation's `batch_days`/`samples_per_batch` work has been an
indirect proxy for a question a real multi-step gradient could answer
directly: does giving the gradient itself more elapsed time change how it
weighs fast fluxes (`osr`, `olr`) against slow ones (`srd`, `lrd`)? Right now
that's inferred indirectly through how the *forward* state evolves between
single-step snapshots. A checkpointed multi-step gradient would let training
see the actual multi-timestep sensitivity directly, which is a structurally
different (and plausibly much more directly useful) lever for the `lrd`
problem than anything tried in this investigation so far.

## The specific experiment once this exists: vary the integration time

Once wired into `calibrate!`/`compute_gradients!` (or a notebook-local copy,
same pattern as `trenberth_ensemble_uncertainty/`'s modified training
function), treat `N` — the number of checkpointed timesteps the gradient
itself spans — as a swept axis, the same way `batch_days` was swept. Compare
raw per-flux equilibrium bias across N values (not a training-metric proxy,
per the standing rule in `project_trenberth_lw_transmissivity_gradscale_fix`
memory), and check specifically whether a longer-integration-time gradient
closes the `lrd`/`srd` gap that batch_days variation alone couldn't.

## Known caveats, before starting

- The upstream example's own comment: "occasionally this gives a
  SegmentationFault (especially on x86 and for large N), but not always" —
  research code, not hardened. Verify stability at the N values actually
  needed before trusting any result built on it.
- Not wired into `compute_gradients!`/`calibrate!` at all yet — different
  call signature, built for a different use case (one-hot temperature/precip
  sensitivity seeds, not the Trenberth flux loss).
- Real dependency gap: the branch it lives on has diverged substantially from
  the SpeedyWeather version this package targets. Integrating this means a
  real SpeedyWeather version bump, not a config change, and needs checking
  compat with everything else this package depends on.
- Cost: checkpointed multi-step AD is not free even with Revolve — more
  expensive than a single-timestep gradient, cost presumably scales with N.
  Not yet benchmarked in this codebase's actual model configuration
  (trunc=31, nlayers=8, the primitive wet model with the full radiation
  parameterization stack) — the upstream example only tests a bare model.
