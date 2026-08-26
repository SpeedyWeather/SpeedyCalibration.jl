# Trenberth Calibration — Strategy

**Main goal: make the full 6-flux Trenberth diagram calibration workable** (all 6 fluxes — OSR,
SRU, SRD, OLR, LRD, LRU — at least as close to target as the untrained default, at true
multi-year equilibrium, not just on the training-batch proxy).

All experiments in this directory share the same training method (online single-timestep Enzyme
AD, `SpeedyCalibration.jl`) and the same `seasonal_cycle=false, daily_cycle=true` setting
(perpetual-equinox — a standard idealized-model approximation of annual-mean forcing). Seasonal
cycle is **not** part of the ladder below — see "Why not seasonal cycle next" at the bottom.

## Status ladder

| Rung | Config | Status | Evidence |
|---|---|---|---|
| 1 | SW-only, 3 fluxes (OSR/SRU/SRD), 15 params | ✅ **Works** — converges cleanly, published in the thesis | `thesis_15param_shortwave.ipynb`; Ch.5 of the thesis |
| 2 | Full, 6 fluxes, 20 params, joint training | ❌ **Broken** — `srd`/`lrd` end up worse than the untrained default at true equilibrium, every config tried | `trenberth_full.ipynb`; extensively re-tested across `batch_days` ∈ {2,3,5,10,30}, every loss reweighting, staged SW→LW training, `fl`-freezing — see `project_trenberth_lw_transmissivity_gradscale_fix` memory |
| 3a | Full, 6 fluxes, **per-block gradient clipping** | ❌ **Done, failed** — `srd` improved (-7.06 vs production's -12.37) but `lrd` got *worse* (+35.56 vs production's +28.50, vs default's +8.81). Confirms the shared clip budget was not the real bottleneck. | `trenberth_perblock_clip/`, 358 batches, full equilibrium validation run |
| 3b | **5 fluxes** (drop `lrd`), 20 params | ❌ **Done, failed — clean negative.** `lrd` (held out) drifted to +55.35, *worse* than under joint training (+28.50) — dropping it from the loss didn't insulate it. The other 5 trained fluxes also came out worse across the board than the 6-flux production run (e.g. `osr` -11.29 vs +1.58, `lru` +7.87 vs +0.89). | `trenberth_5flux_no_lrd/`, 330 batches, full equilibrium validation run |
| 3c | `grad_clip=5.0` sanity check | 🔵 **Confirmed arbitrary, never fixable-scale.** Clips 86-100% of batches at every parameter count tested (n=1 to n=23) — never in the right ballpark. Doesn't explain rungs 3a/3b's failure by itself (3a already tried fixing the budget-sharing and still failed on `lrd`), but `grad_clip≈500` is a cheap follow-up worth its own experiment eventually. | `trenberth_grad_clip_deepdive/` |
| 3d | Swap LW scheme: `JeevanjeeRadiation` instead of `FriersonLongwaveTransmissivity` | ✅ **GO — pilot confirms clean structural fix.** AD-verified: `emissivity_atmosphere` affects `lrd` with **zero** effect on `olr`; `α`/emissivities affect `olr` with **zero** effect on `lrd`. Exactly the decoupling 3a/3b's failures imply is needed, and it's structurally present in this scheme (no shared field, unlike Frierson). | `trenberth_jeevanjee_lw_deepdive/` (pilot only, `trunc=15`) |
| 4 | `trenberth_jeevanjee_full/` — full `trunc=31` calibration swapping in `JeevanjeeRadiation` | ❌ **Done, failed — production-resolution spinup is unstable, not just a tuning problem.** The pilot's decoupling result (3d) only tested 60 steps at `trunc=15`; at `trunc=31` over the full 180-day spinup the model NaNs regardless of `emissivity_atmosphere`: the pilot's default (0.3) survives ~71 real days then NaNs from a slow monotonic cooling drift, a warmer attempt (0.95, tried to fix the cold bias) NaNs within ~9 days from a fast runaway-warming failure instead. Neither end of the range is stable long enough to reach 180-day spinup. Parked per user direction (2026-08-25) — not pursuing further (no bisection, no Jacobian/LP check, no checkpointed gradients). | `trenberth_jeevanjee_full/grad_scale_check.jl`, `spinup_diagnostic.jl` (both NaN points), `emissivity_sweep.jl` |
| 4b | `trenberth_full_cloudabs/` — Frierson scheme (rung 2's method), + 1 new SW param `absorptivity_cloud_base` | ✅ **Real, targeted improvement — best `srd` of the whole investigation.** Untouched otherwise from the rung-2 production config (`batch_days=2`, relative-error weighting, 400 batches, ~23 min wall time). True 7-year-equilibrium bias (this run's own default vs. trained, same baseline): `osr` −15.0→**+5.4**, `sru` −2.8→+5.4 (slightly worse), `srd` +30.6→**+3.4** (best of the investigation), `olr` +18.7→**+5.6**, `lrd` +5.0→+29.5 (same recurring damage — untouched by this change, expected), `lru` +0.4→−0.1. `absorptivity_cloud_limit` (sibling param) confirmed structural zero gradient, stays excluded. **New production baseline going forward.** | `trenberth_full_cloudabs/trenberth_full_cloudabs.jl`, `postprocess.jl` |
| 5 | `trenberth_seasonal/` — rung 4b's config retrained with `seasonal_cycle=true` | ⏳ **In progress (started 2026-08-26).** Real gap found first: `calibrate!` had `seasonal_cycle` hardcoded `false` (not configurable at all) while `run_climate_validation` already defaults to `seasonal_cycle=true` — every prior rung in this table was TRAINED against permanent equinox but VALIDATED against a real seasonal cycle. Added `seasonal_cycle::Bool=false` to `TrainingConfig` (`src/training.jl`), threaded through; warm-started from 4b's `best_params`; spinup lengthened 180→730 days for seasonal phase-lock (unvalidated judgment call, not tested against a longer spinup yet). Proceeding despite this section's own original caution below, per explicit user direction. | `trenberth_seasonal/trenberth_seasonal_full.jl`, `default_baseline.jl` |

## Why rung 2 breaks: the diagnosed root causes

1. **OLR and LRD share one physical lever.** `FriersonLongwaveTransmissivity` uses a single
   transmissivity field for both the upward (OLR) and downward (LRD) radiation beams — confirmed
   directly in SpeedyWeather's source, not inferred. No combination of the 5 LW parameters can
   improve one without moving the other. This is real physics, not a tunable hyperparameter.
2. **The training method's gradient is a short-sighted proxy.** Single-timestep AD, averaged over
   a short batch window, computes ∂L/∂θ from the *instantaneous* state. `SRD`/`LRD` depend on
   slow-responding state (surface/soil temperature, humidity, cloud formation) that hasn't caught
   up to already-applied parameter changes — confirmed by the gradient sign never flipping across
   an entire 240+ batch run (a parameter near a true optimum should show sampling noise flipping
   sign; these don't, ever).
3. Everything that's been tried to work around this — reweighting, freezing the worst-coupled
   parameter, staged/curriculum training, sweeping `batch_days` 2→30 — moves along the same
   trade-off curve or makes things worse. None of it breaks the structural constraint in (1) or
   fixes the short-sightedness in (2).

## Rung 3: what B and D actually showed (both closed out negative)

Both kept everything else (20 params, `batch_days=2`, relative-error flux weighting, training
config) identical to the rung-2 production run in `trenberth_full.ipynb`, changing exactly one
thing each — so their results are directly comparable to the rung-2 baseline and to each other.

- **`trenberth_perblock_clip/` (candidate B) — failed.** Rung 2's shared global-norm gradient
  clip was found to saturate on **100% of batches** — a permanent renormalization, not an
  occasional safety clamp, with the LW-transmissivity block's structurally larger raw gradients
  dominating the shared budget every batch. Giving the SW and LW blocks independent clip budgets
  improved `srd` (-7.06 vs production's -12.37) but made `lrd` *worse* (+35.56 vs production's
  +28.50, vs default's +8.81). The shared-clip-budget was real but not the actual bottleneck.
- **`trenberth_5flux_no_lrd/` (candidate D) — failed, more decisively.** Dropped `lrd` from the
  loss entirely (kept `olr`). Result: `lrd`, tracked as a held-out diagnostic, drifted to +55.35
  — *worse* than under joint 6-flux training (+28.50), purely as a side effect of the LW params
  moving to satisfy `olr`. The coupling drags `lrd` along even when it isn't a target at all. The
  other 5 trained fluxes also came out worse across the board than the 6-flux production run.

**Why this settles the question that motivated rung 3d:** both experiments ruled out "it's an
optimizer/loss-design artifact" — B fixed the clip-budget-sharing mechanism directly and `lrd`
still got worse; D removed all incentive to disturb `lrd` and it still got worse. That leaves
finding (1) — the physically shared transmissivity field — as the real, sufficient explanation,
and points straight at a scheme swap rather than more loss/optimizer tuning. See rung 3d below.

## Rung 3d → 4: the scheme swap (current best lead)

`trenberth_jeevanjee_lw_deepdive/` tested whether `JeevanjeeRadiation` — SpeedyWeather's other
longwave scheme — structurally avoids Frierson's shared-field problem. It computes `lrd` as a
standalone closed-form term (`ϵ·σ·T⁴`, one dedicated parameter `emissivity_atmosphere`) and `olr`
via a completely separate upward-flux accumulation (`α`, `emissivity_ocean`/`land`) — no shared
array between the two, unlike Frierson. Confirmed via real Enzyme AD gradients (not just reading
the source): `d(olr)/d(emissivity_atmosphere) = 0` exactly, `d(lrd)/d(α) = 0` exactly, both
non-zero the *other* way round. Clean decoupling, no zero-gradient parameters, default-parameter
climate in a plausible (if untuned) ballpark. One real caveat found along the way: this scheme is
numerically unstable at this codebase's usual `trunc=5` pilot resolution (matches a warning in
the scheme's own docstring) — use `trunc=15`+ for any further pilot work.

**Current priority: build `trenberth_jeevanjee_full/`**, a full `trunc=31` `calibrate!` run —
same 15 SW/albedo params from `trenberth_full.ipynb`, same relative-error weighting, `batch_days=2`,
LW block replaced with the 4 Jeevanjee parameters validated in the pilot.

## Why not seasonal cycle next

Both the working rung-1 result and the broken rung-2 result share the *same*
`seasonal_cycle=false` setting — it's not a confound between "works" and "broken," it's been held
constant. The diagnosed root causes above are about *which fluxes are targeted* and *how the
gradient is computed*, not about the annual cycle. Turning on seasonal cycle adds another slow
timescale on top of an already poorly-understood slow-timescale problem (finding 2 above) without
testing either current hypothesis — more likely to make `lrd`/`srd` drift worse than to diagnose
anything. Treat "does the fix hold under a real seasonal cycle" as a **later, separate validation
step (rung 4)**, only once rung 3 actually produces a working config — otherwise a seasonal-cycle
run failing wouldn't tell us whether seasonal cycle broke it or the already-known problem did.

## Full history

See the `project_trenberth_lw_transmissivity_gradscale_fix`,
`project_thesis_full_trenberth_sw_lw_coupling`, and `project_cloud_albedo_trenberth_diagnostic`
memory entries for the complete chronological investigation (every batch_days/reweighting/staging
experiment, exact numbers, and corrected-mistake trail). `trenberth_investigation_writeup/` has a
narrative notebook version.
