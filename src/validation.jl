# Internal helper
# Builds a model with params applied but does NOT call initialize!.
# Callers must add any callbacks before calling initialize!(model).
function _build_model(
        param_dict  :: Dict{Symbol,Float32},
        param_specs :: Vector{ParamSpec};
        trunc        :: Int  = 31,
        nlayers      :: Int  = 8,
        daily_cycle  :: Bool = true,
        seasonal_cycle :: Bool = true,
        dt           :: Union{Period,Nothing} = nothing,
        model_kwargs :: NamedTuple = (;),
    )
    sg     = SpectralGrid(trunc=trunc, nlayers=nlayers)
    planet = Earth(sg; daily_cycle=daily_cycle, seasonal_cycle=seasonal_cycle)
    time_stepping = isnothing(dt) ? Leapfrog(sg) : Leapfrog(sg; Δt_at_T31=dt)
    model  = PrimitiveWetModel(sg; planet=planet, time_stepping=time_stepping, model_kwargs...)
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
        seasonal_cycle :: Bool   = false,
        model_kwargs :: NamedTuple = (;),
    )
    model, _ = _build_model(param_dict, param_specs; trunc, nlayers, daily_cycle, seasonal_cycle, model_kwargs)
    sim = initialize!(model)
    sim.variables.prognostic.clock.time = start_date
    return sim
end

"""
    run_climate_validation(result; n_years, stat_years) → NamedTuple

Run a climate validation with `result.best_params` (the lowest-smoothed-loss
checkpoint, not `result.final_params` — training can drift past its optimum
under an unbalanced multi-flux loss, so the best checkpoint is the
representative one to validate). Returns equilibrium means over the last
`stat_years` of a `n_years` run.

Defaults (`n_years=4`, `stat_years=3`, `seasonal_cycle=true`) match the thesis's own
methodology exactly: "Equilibrium values are weighted global means over the final
three years of a four-year T31/L8 simulation with original Fortran SPEEDY default
parameters and the seasonal cycle active." Earlier Trenberth-investigation notebooks
called this with `n_years=7, stat_years=5, seasonal_cycle=false` (perpetual equinox) —
a deliberate, documented departure to keep those specific comparisons internally
consistent (see `examples/README.md`, "Why not seasonal cycle next"), not this
function's own baseline. Pass those explicitly if you need to reproduce that earlier,
non-thesis methodology instead.

Also runs the same simulation with default parameters for comparison.
"""
function run_climate_validation(
        result      :: TrainingResult;
        n_years     :: Int = 4,
        stat_years  :: Int = 3,
        dt          :: Union{Period,Nothing} = nothing,
        seasonal_cycle :: Bool = true,
    )
    cfg   = result.config
    specs = result.param_specs

    default_dict  = Dict{Symbol,Float32}()
    trained_dict  = result.best_params

    default_clm = _run_clm(default_dict,  specs, cfg, n_years, "default"; dt=dt, seasonal_cycle=seasonal_cycle)
    trained_clm = _run_clm(trained_dict,  specs, cfg, n_years, "trained"; dt=dt, seasonal_cycle=seasonal_cycle)

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

# Internal helpers

function _run_clm(param_dict, specs, cfg, n_years, label; dt=nothing, seasonal_cycle=false)
    println("Climate run: $label ...")
    # `hasfield` guard: TrainingConfig objects loaded from result.jld2 files saved before
    # `model_kwargs` was added get reconstructed by JLD2 without that field -- treat as (;).
    mk = hasfield(typeof(cfg), :model_kwargs) ? cfg.model_kwargs : (;)
    model, sg = _build_model(param_dict, specs; trunc=cfg.trunc, nlayers=cfg.nlayers, dt=dt,
                              seasonal_cycle=seasonal_cycle, model_kwargs=mk)
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
