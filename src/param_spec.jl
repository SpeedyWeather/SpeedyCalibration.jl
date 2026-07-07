"""
    ParamSpec(name, path; bounds, initial, grad_scale)

Specifies a single trainable parameter.

- `name`: human-readable identifier (used in history keys and logging)
- `path`: property chain into the model, e.g. `[:shortwave_radiation, :cloud_albedo]`
- `bounds`: physical `(lb, ub)` — enforced structurally via sigmoid reparameterisation
- `initial`: starting value; `nothing` reads the current model default
- `grad_scale`: per-parameter learning-rate multiplier (compensates weak gradients)
"""
struct ParamSpec
    name       :: Symbol
    path       :: Vector{Symbol}
    bounds     :: Tuple{Float32,Float32}
    initial    :: Union{Nothing,Float32}
    grad_scale :: Float32
end

function ParamSpec(name, path;
                   bounds     = (0.01f0, 1.0f0),
                   initial    = nothing,
                   grad_scale = 1.0f0)
    ParamSpec(Symbol(name), collect(Symbol, path),
              Float32.(bounds),
              isnothing(initial) ? nothing : Float32(initial),
              Float32(grad_scale))
end

# ── Property path helpers ─────────────────────────────────────────────────────

function get_by_path(obj, path::Vector{Symbol})
    result = obj
    for p in path
        result = getproperty(result, p)
    end
    return result
end

function set_by_path!(obj, path::Vector{Symbol}, value)
    parent = obj
    for p in path[1:end-1]
        parent = getproperty(parent, p)
    end
    setproperty!(parent, path[end], value)
end

# ── Sigmoid reparameterisation ────────────────────────────────────────────────
# The optimizer operates in unconstrained θ_raw ∈ (-∞,+∞).
# Physical θ_phys ∈ (lb, ub) is recovered via the sigmoid.
# Bounds are therefore structurally enforced and step sizes shrink naturally
# near the limits without clamping (which would break Adam momentum).

"""Convert physical value to unconstrained (logit) space."""
function to_raw(θ_phys::Float32, lb::Float32, ub::Float32; ε::Float32=1f-4)
    p = clamp((θ_phys - lb) / (ub - lb), ε, 1f0 - ε)
    return log(p / (1f0 - p))
end

"""Map unconstrained θ_raw back to physical space."""
sigmoid_param(θ_raw::Float32, lb::Float32, ub::Float32) =
    lb + (ub - lb) / (1f0 + exp(-θ_raw))

"""Chain-rule factor ∂θ_phys/∂θ_raw = (ub−lb)·σ·(1−σ)."""
function sigmoid_grad_factor(θ_raw::Float32, lb::Float32, ub::Float32)
    s = 1f0 / (1f0 + exp(-θ_raw))
    return (ub - lb) * s * (1f0 - s)
end
