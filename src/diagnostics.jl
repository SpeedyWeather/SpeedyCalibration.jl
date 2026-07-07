"""
    check_nans(arr; label="") -> NamedTuple

Count non-finite values (NaN/Inf) in `arr` and summarise the finite range.
Generalises the ad hoc NaN probes used throughout the differentiability notebooks
(e.g. `probe_lw`, `diagnose_longwave_timeseries.jl`) into a reusable check.
"""
function check_nans(arr; label::AbstractString="")
    n_total  = length(arr)
    finite   = filter(isfinite, vec(arr))
    n_nan    = count(isnan, arr)
    n_inf    = count(x -> isinf(x), arr)
    n_bad    = n_total - length(finite)
    return (
        label      = label,
        n_total    = n_total,
        n_nan      = n_nan,
        n_inf      = n_inf,
        frac_bad   = n_total == 0 ? 0.0 : n_bad / n_total,
        min_finite = isempty(finite) ? NaN : minimum(finite),
        max_finite = isempty(finite) ? NaN : maximum(finite),
        ok         = n_bad == 0,
    )
end

"""
    vertical_gradient_anomalies(profile; ratio_threshold=1.5) -> Vector{NamedTuple}

Flag layers where the gradient to the next layer is anomalously large compared
to the gradient one layer further down, i.e. `|Δ(k,k+1)| > ratio_threshold * |Δ(k+1,k+2)|`.
Generalises the layer-1/2 instability check from
`diagnostic_sw_heating_profile.jl` to an arbitrary vertical profile (temperature,
humidity, etc.) and arbitrary number of layers.

Returns one `NamedTuple` per flagged layer with fields
`(layer, delta_here, delta_next, ratio)`.
"""
function vertical_gradient_anomalies(profile::AbstractVector; ratio_threshold::Real=1.5)
    n = length(profile)
    flags = NamedTuple[]
    n < 3 && return flags
    deltas = diff(profile)
    for k in 1:(n-2)
        d_here, d_next = deltas[k], deltas[k+1]
        d_next == 0 && continue
        ratio = abs(d_here) / abs(d_next)
        if ratio > ratio_threshold
            push!(flags, (layer=k, delta_here=d_here, delta_next=d_next, ratio=ratio))
        end
    end
    return flags
end

# NaNWatchCallback

"""
    NaNWatchCallback(; schedule, fields, bound)

Tracks non-finite counts and blow-up magnitudes for one or more grid fields
(by field name on `vars.grid`, e.g. `:temperature`, `:humidity`) over the
course of a simulation. Attach via
`add!(model, :nan_watch => NaNWatchCallback())`.

Generalises the per-step NaN probes in `diagnose_longwave_timeseries.jl` and
`probe_lw` (previously hand-rolled per longwave scheme) into a reusable,
field-agnostic callback.

`bound` is the magnitude beyond which a finite value is still considered an
unphysical "blow-up" (not just a NaN/Inf).
"""
Base.@kwdef mutable struct NaNWatchCallback <: SpeedyWeather.AbstractCallback
    schedule    :: Schedule       = Schedule(every = Hour(1))
    fields      :: Vector{Symbol} = [:temperature, :humidity]
    bound       :: Float64        = 1e3
    counter     :: Int            = 0
    days        :: Vector{Float64} = Float64[]
    n_nonfinite :: Dict{Symbol,Vector{Int}}     = Dict{Symbol,Vector{Int}}()
    n_blowup    :: Dict{Symbol,Vector{Int}}     = Dict{Symbol,Vector{Int}}()
    max_abs     :: Dict{Symbol,Vector{Float64}} = Dict{Symbol,Vector{Float64}}()
end

function SpeedyWeather.initialize!(
        cb::NaNWatchCallback,
        vars::SpeedyWeather.Variables,
        model::SpeedyWeather.AbstractModel)
    initialize!(cb.schedule, vars.prognostic.clock)
    n = cb.schedule.steps
    cb.days = Vector{Float64}(undef, n)
    for f in cb.fields
        cb.n_nonfinite[f] = Vector{Int}(undef, n)
        cb.n_blowup[f]    = Vector{Int}(undef, n)
        cb.max_abs[f]     = Vector{Float64}(undef, n)
    end
    cb.counter = 0
    return nothing
end

function SpeedyWeather.callback!(
        cb::NaNWatchCallback,
        vars::SpeedyWeather.Variables,
        model::SpeedyWeather.AbstractModel)
    isscheduled(cb.schedule, vars.prognostic.clock) || return nothing
    cb.counter += 1
    i  = cb.counter
    t0 = vars.prognostic.clock.start
    cb.days[i] = Dates.value(vars.prognostic.clock.time - t0) / 86_400_000.0

    for f in cb.fields
        data = getfield(vars.grid, f)
        finite_vals = filter(isfinite, vec(Float64.(collect(data))))
        cb.n_nonfinite[f][i] = length(data) - length(finite_vals)
        cb.n_blowup[f][i]    = count(x -> abs(x) > cb.bound, finite_vals)
        cb.max_abs[f][i]     = isempty(finite_vals) ? NaN : maximum(abs.(finite_vals))
    end
    return nothing
end

SpeedyWeather.finalize!(::NaNWatchCallback, args...) = nothing

"""
    any_anomaly(cb::NaNWatchCallback) -> Bool

`true` if any tracked field ever recorded a NaN/Inf or exceeded `cb.bound`.
"""
function any_anomaly(cb::NaNWatchCallback)
    for f in cb.fields
        any(>(0), cb.n_nonfinite[f]) && return true
        any(>(0), cb.n_blowup[f])    && return true
    end
    return false
end
