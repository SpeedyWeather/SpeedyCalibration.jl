# Training

[`calibrate!`](@ref) runs the full online calibration loop and returns a
[`TrainingResult`](@ref). This page explains the `TrainingConfig` fields, the
internal loop structure, and how to interpret the result.

## `TrainingConfig`

All hyperparameters live in one struct so they can be saved alongside results:

```@example training
using SpeedyCalibration

cfg = TrainingConfig(
    spinup_days         = 180,
    batch_days          = 3.0,
    samples_per_batch   = 20,
    max_batches         = 300,
    loss_threshold      = 500f0,
    patience            = 30,
    grad_clip           = 5f0,
    enable_lr_decay     = true,
    lr_decay_factor     = 0.5f0,
    lr_plateau_patience = 50,
    min_lr              = 1f-6,
    max_lr_decays       = 3,
    trunc               = 31,
    nlayers             = 8,
    start_date          = DateTime(2000, 3, 21),
    verbose             = true,
)
```

All fields are keyword arguments with sensible defaults, so you only need to
specify what you want to change.

| Field | Default | Description |
|---|---|---|
| `spinup_days` | 180 | Days to run before collecting any gradients |
| `batch_days` | 3.0 | Simulated days per training batch |
| `samples_per_batch` | 20 | Gradient samples per batch (spread evenly through `batch_days`) |
| `max_batches` | 300 | Hard stop if not converged |
| `loss_threshold` | 500 | Smoothed-loss value at which early stopping triggers |
| `patience` | 30 | Consecutive batches below `loss_threshold` before stopping |
| `grad_clip` | 5.0 | Maximum L2 norm of the gradient vector |
| `enable_lr_decay` | true | Halve the LR when loss plateaus |
| `lr_decay_factor` | 0.5 | Multiplicative factor for LR decay |
| `lr_plateau_patience` | 50 | Batches with no improvement before LR decay |
| `min_lr` | 1e-6 | Floor on the learning rate |
| `max_lr_decays` | 3 | Maximum number of LR halvings |
| `trunc` | 31 | Spectral truncation (T31 ≈ 400 km resolution) |
| `nlayers` | 8 | Number of vertical levels |
| `start_date` | 2000-03-21 | Simulation clock start (March equinox) |
| `verbose` | true | Print progress table during training |

### Quick test config

```@example training
qcfg = quick_test_config()   # trunc=5, nlayers=3, spinup=5d, max_batches=5
```

Use `quick_test_config()` before a long run to verify that your parameter paths,
loss config, and bounds are set up correctly without waiting hours for a result.
Additional keyword arguments are forwarded to `TrainingConfig`:

```julia
quick_test_config(samples_per_batch=5)
```

## The training loop

The loop structure inside `calibrate!` is:

```
1. Build model, apply initial parameter values, initialise simulation
2. Convert parameters to unconstrained (logit) space
3. Spinup: advance spinup_days × steps_per_day timesteps
4. For each batch:
   a. For each sample in 1:samples_per_batch:
      - Advance steps_per_sample timesteps
      - Call compute_gradients!(variables, model, loss_config, param_specs)
        (Enzyme backward pass; state is saved and restored)
      - Accumulate valid gradients and flux means
   b. Average accumulated gradients
   c. Apply grad_scale and sigmoid chain-rule factor per parameter
   d. Clip gradient vector to grad_clip L2 norm
   e. Optimisers.update! in unconstrained space
   f. Convert back to physical space via sigmoid
   g. SpeedyWeather.reconstruct(model, new_params)
   h. Record history
   i. Check LR decay and early stopping
```

The key insight is in step 4a: the Enzyme backward pass operates on a *single
timestep* and the state is fully restored afterwards. The forward simulation
therefore continues without interruption, but every `steps_per_sample` steps we
also extract a gradient estimate. Averaging 20 such estimates per batch filters
out the per-timestep noise introduced by chaotic dynamics.

## Choosing hyperparameters

**Spinup.** The model needs to reach a statistically stationary state before
gradients are meaningful. For T31 with 8 levels, 180 days is conservative but
safe. You can reduce this to 30–60 days if you know the model equilibrates quickly
for your parameter configuration.

**Batch size.** `batch_days=3.0, samples_per_batch=20` means one gradient sample
every ∼3.6 hours of simulated time. Reducing `batch_days` collects gradients more
frequently (larger effective batch size per wall-clock time) but each sample is
more correlated with the previous one. 1–3 days is a reasonable range.

**Learning rate.** Start with `Optimisers.Adam(5f-3)`. If the loss oscillates
wildly, reduce to `1f-3`. If convergence is very slow after LR decay, try
`1f-2`. The sigmoid reparameterisation means the effective physical-space step
size already shrinks near bounds, so a single LR can work across parameters with
very different scales.

**Loss threshold.** This is problem-dependent. For `TRENBERTH_LOSS` with equal
weights and the default targets, a well-converged run typically reaches ∼50–200;
the default of 500 is intentionally conservative. Inspect `result.history[:smoothed_loss]`
after a first run and set the threshold accordingly.

## The `TrainingResult`

```@example training
using Optimisers, SpeedyWeather

params = [ParamSpec(:cloud_albedo,
                    [:shortwave_radiation, :clouds, :cloud_albedo];
                    bounds=(0.1f0, 0.85f0))]

result = calibrate!(params, Optimisers.Adam(1f-2), OSR_LOSS, quick_test_config())
```

The result bundles everything:

```@example training
result.final_params       # Dict{Symbol,Float32}: name → trained value
result.conv_info          # NamedTuple with convergence summary
result.config             # the TrainingConfig used
result.loss_config        # the LossConfig used
result.param_specs        # the Vector{ParamSpec} used
```

```@example training
result.conv_info.stop_reason
result.conv_info.best_smoothed_loss
result.conv_info.total_time
```

### History dictionary

`result.history` is a `Dict{Symbol, Vector}` with one entry per scalar per batch:

| Key | Content |
|---|---|
| `:batch` | batch index |
| `:loss` | instantaneous batch loss |
| `:smoothed_loss` | rolling mean over last `loss_window_size` batches |
| `:lr` | current learning rate |
| `:elapsed_time` | wall-clock seconds since start |
| `:param_change` | mean relative parameter change from previous batch |
| `:<flux_key>` | e.g. `:osr`, `:olr`: batch mean of that flux |
| `:<name>` | e.g. `:cloud_albedo`: parameter value after the batch update |
| `:grad_<name>` | mean Enzyme gradient for that parameter |
| `:gradstd_<name>` | std of Enzyme gradient across the batch samples |

All vectors have the same length (number of completed batches).

## Saving and resuming

Save a result to JLD2 before a long validation run or before closing Julia:

```julia
save_result(result, "run_trenberth_v1.jld2")
```

Load it back later:

```julia
result = load_result("run_trenberth_v1.jld2")
```

!!! note "Resume-safety"
    The JLD2 file stores the full `TrainingResult` struct including `config`,
    `loss_config`, and `param_specs`, so the file is self-contained. You do not
    need to re-define any parameter specs to load and analyse a saved result.

## Monitoring gradients

The gradient standard deviation across samples (`gradstd_<name>`) tells you how
noisy the per-timestep gradient estimate is for each parameter. A high
`gradstd / |grad_mean|` ratio (signal-to-noise < 1) means the averaging is not
sufficient; increase `samples_per_batch`. Values much larger than the mean are
normal for chaotic systems; the mean still converges to the correct sensitivity
as long as the noise is unbiased.
