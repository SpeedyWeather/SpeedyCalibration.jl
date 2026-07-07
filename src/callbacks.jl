"""
    DailyMeansCallback{NF}

Records area-weighted daily mean diagnostics throughout a simulation.
Attach to a model via `add!(model, :daily_means => DailyMeansCallback(spectral_grid))`.

Tracked fields: temperature profile, OLR, OSR, SRD, SRU, LRD, LRU,
total/convective/large-scale precipitation, cloud top.
"""
Base.@kwdef mutable struct DailyMeansCallback{NF} <: SpeedyWeather.AbstractCallback
    schedule      :: Schedule       = Schedule(every = Day(1))
    counter       :: Int            = 0
    days          :: Vector{Float64} = Float64[]
    temp          :: Matrix{NF}     = zeros(NF, 0, 0)   # (time × layer)
    olr           :: Vector{NF}     = NF[]
    osr           :: Vector{NF}     = NF[]
    srd           :: Vector{NF}     = NF[]
    sru           :: Vector{NF}     = NF[]
    lrd           :: Vector{NF}     = NF[]
    lru           :: Vector{NF}     = NF[]
    precip_total  :: Vector{NF}     = NF[]
    precip_conv   :: Vector{NF}     = NF[]
    precip_ls     :: Vector{NF}     = NF[]
    cloud_top     :: Vector{NF}     = NF[]
end

DailyMeansCallback(sg::SpectralGrid) = DailyMeansCallback{sg.NF}()

function SpeedyWeather.initialize!(
        cb::DailyMeansCallback{NF},
        vars::SpeedyWeather.Variables,
        model::SpeedyWeather.AbstractModel) where NF
    initialize!(cb.schedule, vars.prognostic.clock)
    n       = cb.schedule.steps
    nlayers = model.geometry.nlayers
    cb.days = Vector{Float64}(undef, n)
    cb.temp = Matrix{NF}(undef, n, nlayers)
    for v in (:olr, :osr, :srd, :sru, :lrd, :lru,
              :precip_total, :precip_conv, :precip_ls, :cloud_top)
        setfield!(cb, v, Vector{NF}(undef, n))
    end
    cb.counter = 0
    return nothing
end

function SpeedyWeather.callback!(
        cb::DailyMeansCallback{NF},
        vars::SpeedyWeather.Variables,
        model::SpeedyWeather.AbstractModel) where NF
    isscheduled(cb.schedule, vars.prognostic.clock) || return nothing
    cb.counter += 1
    i = cb.counter
    t0 = vars.prognostic.clock.start
    cb.days[i] = Dates.value(vars.prognostic.clock.time - t0) / 86_400_000.0
    cb.temp[i, :] .= vars.grid.temp_average
    p = vars.parameterizations
    cb.olr[i] = _cosine_mean(p.outgoing_longwave,      model)
    cb.osr[i] = _cosine_mean(p.outgoing_shortwave,     model)
    cb.srd[i] = _cosine_mean(p.surface_shortwave_down, model)
    cb.sru[i] = _cosine_mean(p.surface_shortwave_up,   model)
    cb.lrd[i] = _cosine_mean(p.surface_longwave_down,  model)
    cb.lru[i] = _cosine_mean(p.surface_longwave_up,    model)
    scale = 86_400f0 * 1000f0
    cb.precip_total[i] = _cosine_mean(p.rain_rate,             model) * scale
    cb.precip_conv[i]  = _cosine_mean(p.rain_rate_convection,  model) * scale
    cb.precip_ls[i]    = _cosine_mean(p.rain_rate_large_scale, model) * scale
    cb.cloud_top[i]    = mean(Float32.(collect(p.cloud_top)))
    return nothing
end

SpeedyWeather.finalize!(::DailyMeansCallback, args...) = nothing

_cosine_mean(field, model::SpeedyWeather.AbstractModel) =
    Float32(sum(RingGrids.zonal_mean(field) .* model.geometry.coslat) /
            sum(model.geometry.coslat))
