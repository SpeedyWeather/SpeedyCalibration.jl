# Loss configuration

The [`LossConfig`](@ref) struct defines the loss function: which radiative fluxes to
track, their observational reference values, and how much weight each flux contributes
to the scalar loss. SpeedyCalibration ships three ready-made configurations and makes
it easy to define your own.

## Built-in configurations

All three are based on the global-mean energy budget from
[Trenberth et al. (2009)](https://doi.org/10.1175/2008BAMS2634.1). They are
exported as named constants:

### `OSR_LOSS`

Single-flux loss targeting outgoing shortwave radiation (OSR).

| Flux | Target [W m⁻²] | Weight |
|------|---------------|--------|
| OSR  | 101.9         | 1.0    |

Good for quickly verifying that a parameter (e.g. `cloud_albedo`) has a non-zero
gradient and that the training loop works.

### `OSR_SRU_SRD_LOSS`

Three-flux shortwave budget loss.

| Flux | Target [W m⁻²] | Weight |
|------|---------------|--------|
| OSR (outgoing SW at TOA)       | 101.9 | 1.0 |
| SRU (surface SW up)            |  23.0 | 1.0 |
| SRD (surface SW down)          | 168.0 | 1.0 |

Closing the SW budget is a natural first calibration target because SW parameters
(cloud albedo, ozone, surface albedo) have clear, large gradients.

### `TRENBERTH_LOSS`

Six-flux loss covering both the shortwave and longwave budget.

| Flux | Target [W m⁻²] | Weight |
|------|---------------|--------|
| OSR (outgoing SW at TOA)       | 101.9 | 1.0 |
| SRU (surface SW up)            |  23.0 | 0.5 |
| SRD (surface SW down)          | 168.0 | 0.5 |
| OLR (outgoing LW at TOA)       | 235.0 | 1.0 |
| LRD (surface LW down)          | 333.0 | 0.3 |
| LRU (surface LW up)            | 398.0 | 0.3 |

The weights reflect confidence in the observational reference and the sensitivity
of the corresponding parameters. OLR and OSR are the most tightly constrained
satellite observations; surface LW fluxes carry larger uncertainty so they
receive lower weight.

!!! note "Latent and sensible heat"
    The Trenberth diagram also lists latent heat (LH ≈ 80 W m⁻²) and sensible heat
    (SH ≈ 17 W m⁻²). These are not yet included in `TRENBERTH_LOSS` because their
    field paths require separate handling (precipitation rate → energy flux conversion).
    A future `TRENBERTH_FULL_LOSS` will include them.

## Creating a custom `LossConfig`

```julia
using SpeedyCalibration

# Target only OLR: useful when calibrating LW parameters
olr_only = LossConfig(
    [:olr];
    targets = Dict(:olr => 235.0f0),
    weights = Dict(:olr => 1.0f0),
)
```

The `flux_keys` must be a subset of the six supported keys: `:osr`, `:sru`, `:srd`,
`:olr`, `:lrd`, `:lru`. Passing an unknown key throws an error immediately.

```julia
# SW + LW balance at TOA only
toa_balance = LossConfig(
    [:osr, :olr];
    targets = Dict(:osr => 101.9f0, :olr => 235.0f0),
    weights = Dict(:osr =>   1.0f0, :olr =>   1.0f0),
)
```

## How the loss is computed

Each training sample computes area-weighted global means using Gaussian quadrature
weights, which are exact for SpeedyWeather's `OctahedralGaussianGrid`. For a flux
key k:

```math
\bar{F}_k = \frac{\sum_j w_j \bar{F}_{k,j}}{\sum_j w_j}
```

where the sum is over latitude rings j, w_j are the Gaussian weights, and
$\bar{F}_{k,j}$ is the zonal mean of flux k on ring j (finite values only).

The loss is then:

```math
L = \sum_k \lambda_k \left(\bar{F}_k - F_k^*\right)^2
```

where λ_k is `weights[k]` and F_k* is `targets[k]`. This is a weighted mean-squared
error in physical units (W m⁻²)².

## Choosing weights

Weights serve two purposes:

1. **Physical normalisation.** The absolute value of (F̄_k − F_k*)² depends on the
   scale of the flux. OSR deviations of ∼10 W m⁻² are much more common than SRU
   deviations of the same size; the two fluxes occupy different parts of the energy
   budget. Equal weights therefore give the larger fluxes proportionally more
   influence. Normalising each term by (F_k*)² makes the loss dimensionless with
   equal relative deviations treated equally.

2. **Gradient signal balance.** If one flux has a very strong gradient with respect
   to the trainable parameters (e.g. OSR for `cloud_albedo`) and another a very
   weak one (e.g. LRU for the same parameter), a high weight on the weak-gradient
   term adds noise without useful signal. Reduce such weights or exclude the flux.

!!! tip "Iterative refinement"
    A practical workflow is to start with `OSR_LOSS`, check which other fluxes
    improve or degrade, then augment the loss to penalise the problematic ones.
    Jumping straight to `TRENBERTH_LOSS` with many parameters and a large learning
    rate often leads to oscillations.
