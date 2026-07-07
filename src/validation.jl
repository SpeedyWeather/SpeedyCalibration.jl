# ── Internal helper ───────────────────────────────────────────────────────────
# Builds a model with params applied but does NOT call initialize!.
# Callers must add any callbacks before calling initialize!(model).
function _build_model(
        param_dict  :: Dict{Symbol,Float32},
        param_specs :: Vector{ParamSpec};
        trunc       :: Int  = 31,
        nlayers     :: Int  = 8,
        daily_cycle :: Bool = true,
        dt          :: Union{Period,Nothing} = nothing,
    )
    sg     = SpectralGrid(trunc=trunc, nlayers=nlayers)
    planet = Earth(sg; daily_cycle=daily_cycle, seasonal_cycle=false)
    time_stepping = isnothing(dt) ? Leapfrog(sg) : Leapfrog(sg; Δt_at_T31=dt)
    model  = PrimitiveWetModel(sg; planet=planet, time_stepping=time_stepping)
    p      = vec(parameters(model))
    for spec in param_specs
        haskey(param_dict, spec.name) && set_by_path!(p, spec.path, param_dict[spec.name])
    end
    return SpeedyWeather.reconstruct(model, p), sg
end

"""
    build_climate_sim(param_dict, param_specs; trunc, nlayers, start_date, daily_cycle) → Simulation

Construct a fresh simulation with parameter values from `param_dict`
(a `Dict{Symbol,Float32}` keyed by `ParamSpec.name`).
"""
function build_climate_sim(
        param_dict   :: Dict{Symbol,Float32},
        param_specs  :: Vector{ParamSpec};
        trunc        :: Int      = 31,
        nlayers      :: Int      = 8,
        start_date   :: DateTime = DateTime(2000, 3, 21),
        daily_cycle  :: Bool     = true,
    )
    model, _ = _build_model(param_dict, param_specs; trunc, nlayers, daily_cycle)
    sim = initialize!(model)
    sim.variables.prognostic.clock.time = start_date
    return sim
end

"""
    run_climate_validation(result; n_years, stat_years) → NamedTuple

Run a climate validation with the trained parameter values from `result`.
Returns equilibrium means over the last `stat_years` of a `n_years` run.

Also runs the same simulation with default parameters for comparison.
"""
function run_climate_validation(
        result      :: TrainingResult;
        n_years     :: Int = 7,
        stat_years  :: Int = 5,
        dt          :: Union{Period,Nothing} = nothing,
    )
    cfg   = result.config
    specs = result.param_specs

    default_dict  = Dict{Symbol,Float32}()
    trained_dict  = result.final_params

    default_clm = _run_clm(default_dict,  specs, cfg, n_years, "default"; dt=dt)
    trained_clm = _run_clm(trained_dict,  specs, cfg, n_years, "trained"; dt=dt)

    stat_start = n_years - stat_years + 1
    return (
        default = _equilibrium_stats(default_clm, stat_start, n_years),
        trained = _equilibrium_stats(trained_clm, stat_start, n_years),
        n_years     = n_years,
        stat_years  = stat_years,
        default_cb  = default_clm,
        trained_cb  = trained_clm,
    )
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _run_clm(param_dict, specs, cfg, n_years, label; dt=nothing)
    println("Climate run: $label ...")
    model, sg = _build_model(param_dict, specs; trunc=cfg.trunc, nlayers=cfg.nlayers, dt=dt)
    cb = DailyMeansCallback(sg)
    add!(model, :daily_means => cb)
    sim = initialize!(model)
    sim.variables.prognostic.clock.time = cfg.start_date
    run!(sim; period=Day(n_years * 365), output=false)
    return cb
end

function _equilibrium_stats(cb::DailyMeansCallback, year_start, year_end)
    i0 = searchsortedfirst(cb.days, Float64((year_start - 1) * 365))
    i1 = searchsortedlast(cb.days,  Float64(year_end * 365))
    idx = i0:i1
    isempty(idx) && (idx = eachindex(cb.osr))
    return (
        osr          = mean(cb.osr[idx]),
        olr          = mean(cb.olr[idx]),
        srd          = mean(cb.srd[idx]),
        sru          = mean(cb.sru[idx]),
        lrd          = mean(cb.lrd[idx]),
        lru          = mean(cb.lru[idx]),
        precip_total = mean(cb.precip_total[idx]),
        temp_profile = vec(mean(cb.temp[idx, :], dims=1)),
    )
end
