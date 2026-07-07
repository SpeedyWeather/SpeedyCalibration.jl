# Defining parameters

The [`ParamSpec`](@ref) struct tells SpeedyCalibration which model parameter to tune,
where to find it, what its physical limits are, and how to weight its gradient.

## The `ParamSpec` struct

```julia
struct ParamSpec
    name       :: Symbol
    path       :: Vector{Symbol}
    bounds     :: Tuple{Float32,Float32}
    initial    :: Union{Nothing,Float32}
    grad_scale :: Float32
end
```

Create a `ParamSpec` with the keyword constructor:

```@example params
using SpeedyCalibration

spec = ParamSpec(:cloud_albedo,
                 [:shortwave_radiation, :clouds, :cloud_albedo];
                 bounds    = (0.1f0, 0.85f0),
                 initial   = 0.35f0,
                 grad_scale = 1.0f0)
```

| Field | Required | Description |
|---|---|---|
| `name` | yes | Identifier used in history keys and log output |
| `path` | yes | Property path from `model` to the parameter |
| `bounds` | no | Physical `(lb, ub)`; default `(0.01, 1.0)` |
| `initial` | no | Starting value; `nothing` reads the current model default |
| `grad_scale` | no | Multiplier on the Enzyme gradient; default `1.0` |

## Finding parameter paths

The `path` is simply the chain of `getproperty` calls that leads from the model to
the scalar parameter. For `cloud_albedo`:

```julia
model.shortwave_radiation.clouds.cloud_albedo
# path = [:shortwave_radiation, :clouds, :cloud_albedo]
```

The easiest way to discover paths is SpeedyWeather's `parameters` function, which
lists every `@param`-annotated field in the model as a table:

```julia
using SpeedyWeather

sg    = SpectralGrid(trunc=31, nlayers=8)
model = PrimitiveWetModel(sg)
params = parameters(model)   # prints a ParameterTable
```

The `fieldname` and `component` columns tell you the last two elements of the path.
Most radiation and convection parameters have three-element paths of the form
`[component, subcomponent, fieldname]`.

!!! note "Parameters with zero gradient"
    Some parameters cannot be trained because their gradient through the AD graph is
    structurally zero. Known examples:
    - `conv_time_scale`: the `Second(time_scale).value` integer conversion is opaque to Enzyme
    - `lsc_rh_threshold`: a step-function boundary; gradient is zero almost everywhere

    Leave these out of your `param_specs` list.

## Physical bounds and sigmoid reparameterisation

Bounds are enforced *structurally*, not by clamping. SpeedyCalibration maps each
physical parameter θ ∈ (lb, ub) to an unconstrained raw value θ_raw ∈ (−∞, +∞)
via the logit transformation:

```math
\theta_\text{raw} = \log\!\left(\frac{p}{1-p}\right), \quad p = \frac{\theta - \text{lb}}{\text{ub} - \text{lb}}
```

The optimizer (Adam, etc.) works in θ_raw space. To recover the physical value:

```math
\theta = \text{lb} + (\text{ub} - \text{lb}) \cdot \sigma(\theta_\text{raw})
```

where σ is the sigmoid function. Because σ saturates near 0 and 1, the effective
step size naturally shrinks as a parameter approaches its physical limit; no
clamping or projected gradient is needed. Gradients from Enzyme (which are
∂L/∂θ_phys) are multiplied by the chain-rule factor ∂θ_phys/∂θ_raw before
the optimizer step.

The choice of bounds therefore matters: too narrow and the optimizer cannot explore
enough; too wide and the sigmoid becomes nearly linear everywhere, losing the
protective saturation.

## The `grad_scale` field

Some parameters have Enzyme gradients that are many orders of magnitude smaller than
others, not because they are unimportant, but because their numerical range is very
different. `grad_scale` is a per-parameter multiplier applied *after* the sigmoid
chain-rule factor, before gradient clipping. It acts as a per-parameter effective
learning rate.

For example, `temp_tropopause` is measured in Kelvin (∼200), so a 1 W/m² change in
flux corresponds to a much smaller gradient than for `cloud_albedo` (∼0.5). Setting
`grad_scale=0.01` for `temp_tropopause` brings it into the same order of magnitude as
the others so that the global gradient clip and Adam's step sizes apply consistently.

Similarly, `entrainment_rate` is O(10⁻⁴); `grad_scale=1000` compensates.

!!! tip "Choosing `grad_scale`"
    Run 10–20 training batches, inspect `result.history[:grad_<name>]` for each
    parameter, and set `grad_scale` so that the resulting pre-clip gradient magnitudes
    are within a factor of ∼10 of each other.

## Defining multiple parameters

Pass a `Vector{ParamSpec}` to `calibrate!`. Parameters can span different
parameterisation modules:

```julia
param_specs = [
    # SW cloud reflection
    ParamSpec(:cloud_albedo,
              [:shortwave_radiation, :clouds, :cloud_albedo];
              bounds=(0.25f0, 0.95f0), initial=0.60f0),
    ParamSpec(:stratocumulus_albedo,
              [:shortwave_radiation, :clouds, :stratocumulus_albedo];
              bounds=(0.10f0, 0.90f0), initial=0.50f0),

    # SW absorption: note the nesting, transmissivity is a sub-struct
    ParamSpec(:absorptivity_water_vapor,
              [:shortwave_radiation, :transmissivity, :absorptivity_water_vapor];
              bounds=(60f0, 140f0), initial=75f0, grad_scale=0.01f0),
    ParamSpec(:ozone_absorption,
              [:shortwave_radiation, :radiative_transfer, :ozone_absorption];
              bounds=(0.002f0, 0.020f0), initial=0.01f0),

    # Surface albedo: lives under :albedo, not :shortwave_radiation
    ParamSpec(:albedo_land,
              [:albedo, :land, :albedo_land];
              bounds=(0.10f0, 0.70f0), initial=0.40f0),

    # LW: Frierson transmissivity parameters
    ParamSpec(:tau0_equator,
              [:longwave_radiation, :transmissivity, :τ₀_equator];
              bounds=(2f0, 12f0), initial=6f0),
    ParamSpec(:emissivity_ocean,
              [:longwave_radiation, :radiative_transfer, :emissivity_ocean];
              bounds=(0.80f0, 1.00f0), initial=0.98f0),
]
```

There is no limit on the number of parameters, but beyond ∼20 the signal-to-noise
ratio in each gradient estimate typically starts to degrade. In that regime, increase
`samples_per_batch` in [`TrainingConfig`](@ref) to compensate.

## Reading the initial model defaults

If you want to start from the SpeedyWeather defaults rather than specifying `initial`
explicitly, set `initial=nothing` (the default). SpeedyCalibration will read the
current parameter value from the model at initialisation time.

You can also check what a model's defaults are before building a `ParamSpec`:

```julia
using SpeedyWeather

sg    = SpectralGrid(trunc=31, nlayers=8)
model = PrimitiveWetModel(sg)

# read cloud_albedo directly
model.shortwave_radiation.clouds.cloud_albedo
```
