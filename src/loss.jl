# Maps flux keys to the field name inside `variables.parameterizations`
const FLUX_FIELD = Dict{Symbol,Symbol}(
    :osr => :outgoing_shortwave,
    :sru => :surface_shortwave_up,
    :srd => :surface_shortwave_down,
    :olr => :outgoing_longwave,
    :lrd => :surface_longwave_down,
    :lru => :surface_longwave_up,
)

"""
    LossConfig(flux_keys; targets, weights)

Defines a weighted-MSE loss over a set of global-mean radiative fluxes.

`flux_keys` must be a subset of `$(collect(keys(FLUX_FIELD)))`.

# Example
```julia
cfg = LossConfig([:osr, :sru];
                 targets = Dict(:osr => 101.9f0, :sru => 23.0f0),
                 weights = Dict(:osr => 1f0,     :sru => 1f0))
```
"""
struct LossConfig
    flux_keys :: Vector{Symbol}
    targets   :: Dict{Symbol,Float32}
    weights   :: Dict{Symbol,Float32}
end

function LossConfig(flux_keys::AbstractVector{Symbol};
                    targets::Dict{Symbol,Float32},
                    weights::Dict{Symbol,Float32})
    for k in flux_keys
        k in keys(FLUX_FIELD) || error("Unknown flux key :$k. Known: $(keys(FLUX_FIELD))")
        haskey(targets, k)    || error("Missing target for :$k")
        haskey(weights, k)    || error("Missing weight for :$k")
    end
    LossConfig(collect(flux_keys), targets, weights)
end

# Pre-built configurations

const OSR_LOSS = LossConfig(
    [:osr];
    targets = Dict(:osr => 101.9f0),
    weights = Dict(:osr => 1f0),
)

const OSR_SRU_SRD_LOSS = LossConfig(
    [:osr, :sru, :srd];
    targets = Dict(:osr => 101.9f0, :sru => 23.0f0, :srd => 168.0f0),
    weights = Dict(:osr => 1f0,     :sru => 1f0,     :srd => 1f0),
)

const TRENBERTH_LOSS = LossConfig(
    [:osr, :sru, :srd, :olr, :lrd, :lru];
    targets = Dict(:osr => 101.9f0, :sru =>  23.0f0, :srd => 168.0f0,
                   :olr => 235.0f0, :lrd => 333.0f0, :lru => 398.0f0),
    weights = Dict(:osr => 1.0f0,   :sru =>  0.5f0,  :srd => 0.5f0,
                   :olr => 1.0f0,   :lrd =>  0.3f0,  :lru => 0.3f0),
)

"""
    make_normalized_loss(flux_keys, targets, initial_means) → LossConfig

Build a LossConfig whose weights are inversely proportional to the squared
initial residuals, so every flux contributes **equally** to the total loss at
the start of training regardless of its absolute scale.

    w_k = 1 / (initial_means[k] - targets[k])²

If a flux has zero initial residual the weight is set to 1.0 (safe fallback).

# Example
```julia
initial = Dict(:osr => 74.5f0, :sru => 18.3f0, :srd => 163.0f0)
targets = Dict(:osr => 101.9f0, :sru => 23.0f0, :srd => 168.0f0)
cfg = make_normalized_loss([:osr, :sru, :srd], targets, initial)
```
"""
function make_normalized_loss(flux_keys::AbstractVector{Symbol},
                               targets::Dict{Symbol,Float32},
                               initial_means::Dict{Symbol,Float32})
    weights = Dict{Symbol,Float32}()
    for k in flux_keys
        r = initial_means[k] - targets[k]
        weights[k] = abs(r) < 1f-6 ? 1f0 : 1f0 / r^2
    end
    LossConfig(collect(flux_keys), targets, weights)
end

# Flux extraction

"""
    compute_flux_means(variables, flux_keys) → Dict{Symbol,Float32}

Area-weighted (Gaussian quadrature) global means for each requested flux.
Returns `nothing` for a key if the field is entirely non-finite.
"""
function compute_flux_means(variables, flux_keys::AbstractVector{Symbol})
    param = variables.parameterizations
    first_field = getproperty(param, FLUX_FIELD[flux_keys[1]])
    _grid  = first_field.grid
    _gw    = Float32.(RingGrids.gaussian_weights(_grid.nlat_half))
    _rings = eachring(_grid)

    totals = Dict{Symbol,Float32}(k => 0f0 for k in flux_keys)
    wsums  = Dict{Symbol,Float32}(k => 0f0 for k in flux_keys)

    for (j, ring) in enumerate(_rings)
        wj = _gw[j]
        for k in flux_keys
            field = getproperty(param, FLUX_FIELD[k])
            v = filter(isfinite, field[ring])
            if !isempty(v)
                totals[k] += wj * Float32(mean(v))
                wsums[k]  += wj
            end
        end
    end

    return Dict{Symbol,Float32}(
        k => (wsums[k] > 0f0 ? totals[k] / wsums[k] : NaN32) for k in flux_keys
    ), wsums
end

"""
    compute_loss(means, loss_config) → Float32

Weighted MSE: `∑_k w_k · (mean_k − target_k)²`.
Returns `Inf32` if any mean is non-finite.
"""
function compute_loss(means::Dict{Symbol,Float32}, cfg::LossConfig)
    any(!isfinite(means[k]) for k in cfg.flux_keys) && return Inf32
    return sum(cfg.weights[k] * (means[k] - cfg.targets[k])^2 for k in cfg.flux_keys)
end

"""
    loss_coefficients(means, loss_config) → Dict{Symbol,Float32}

Returns `2·w_k·(mean_k − target_k)` for each flux: the cotangent seeds
before dividing by area weights in the backward pass.
"""
function loss_coefficients(means::Dict{Symbol,Float32}, cfg::LossConfig)
    return Dict{Symbol,Float32}(
        k => 2f0 * cfg.weights[k] * (means[k] - cfg.targets[k]) for k in cfg.flux_keys
    )
end
