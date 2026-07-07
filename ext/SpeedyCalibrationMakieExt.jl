module SpeedyCalibrationMakieExt

using SpeedyCalibration
using CairoMakie
using GeoMakie
using RingGrids
import GeoMakie.Makie.GeometryBasics: Polygon, Point

# plot_training

function SpeedyCalibration.plot_training(
        result::TrainingResult;
        save_dir::Union{Nothing,AbstractString} = nothing,
    )
    h = result.history
    batches = h[:batch]
    specs   = result.param_specs
    fkeys   = result.loss_config.flux_keys

    ax_kw = (xgridvisible=false, ygridvisible=false,
             topspinevisible=false, rightspinevisible=false)

    # Loss curve
    fig_loss = Figure(size=(900, 320), fontsize=13)
    ax1 = Axis(fig_loss[1,1]; xlabel="Batch", ylabel="Loss",
               title="Training loss", ax_kw...)
    lines!(ax1, batches, h[:loss];          color=:lightgray, label="batch")
    lines!(ax1, batches, h[:smoothed_loss]; color=:royalblue, linewidth=2, label="smoothed")
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig_loss[1,2]; xlabel="Batch", ylabel="Learning rate",
               title="LR schedule", ax_kw...)
    lines!(ax2, batches, h[:lr]; color=:tomato, linewidth=2)

    # Flux trajectories
    ncols   = min(3, length(fkeys))
    nrows   = ceil(Int, length(fkeys) / ncols)
    fig_flux = Figure(size=(ncols*380, nrows*260), fontsize=13)
    for (idx, k) in enumerate(fkeys)
        row = (idx-1) ÷ ncols + 1
        col = (idx-1) % ncols + 1
        ax = Axis(fig_flux[row, col]; xlabel="Batch",
                  ylabel="W m⁻²", title=string(k), ax_kw...)
        lines!(ax, batches, h[k]; color=:royalblue, linewidth=2)
        target = result.loss_config.targets[k]
        hlines!(ax, [target]; color=:black, linestyle=:dash, linewidth=1)
    end

    # Parameter trajectories
    ncols_p  = min(4, length(specs))
    nrows_p  = ceil(Int, length(specs) / ncols_p)
    fig_params = Figure(size=(ncols_p*300, nrows_p*240), fontsize=13)
    for (i, spec) in enumerate(specs)
        row = (i-1) ÷ ncols_p + 1
        col = (i-1) % ncols_p + 1
        ax = Axis(fig_params[row, col]; xlabel="Batch",
                  title=string(spec.name), ax_kw...)
        lines!(ax, batches, h[spec.name]; color=:darkorange, linewidth=2)
        lb, ub = spec.bounds
        hlines!(ax, [lb, ub]; color=:gray60, linestyle=:dash, linewidth=1)
    end

    # Gradient magnitudes
    fig_grads = Figure(size=(ncols_p*300, nrows_p*240), fontsize=13)
    for (i, spec) in enumerate(specs)
        row = (i-1) ÷ ncols_p + 1
        col = (i-1) % ncols_p + 1
        ax = Axis(fig_grads[row, col]; xlabel="Batch",
                  title="∂L/∂$(spec.name)", ax_kw...)
        gk = Symbol("grad_", spec.name)
        sk = Symbol("gradstd_", spec.name)
        lines!(ax, batches, h[gk]; color=:mediumseagreen, linewidth=2)
        if haskey(h, sk)
            band!(ax, batches, h[gk] .- h[sk], h[gk] .+ h[sk];
                  color=(:mediumseagreen, 0.2))
        end
        hlines!(ax, [0f0]; color=:black, linewidth=1)
    end

    if !isnothing(save_dir)
        mkpath(save_dir)
        save(joinpath(save_dir, "fig_loss.pdf"),   fig_loss)
        save(joinpath(save_dir, "fig_flux.pdf"),   fig_flux)
        save(joinpath(save_dir, "fig_params.pdf"), fig_params)
        save(joinpath(save_dir, "fig_grads.pdf"),  fig_grads)
    end

    return (; fig_loss, fig_flux, fig_params, fig_grads)
end

# plot_climate

function SpeedyCalibration.plot_climate(
        validation;
        save_dir::Union{Nothing,AbstractString} = nothing,
        loss_config::Union{Nothing,LossConfig}  = nothing,
    )
    ax_kw = (xgridvisible=false, ygridvisible=false,
             topspinevisible=false, rightspinevisible=false)

    default_cb = validation.default_cb
    trained_cb = validation.trained_cb

    # SW radiation budget
    sw_panels = [
        ("OSR [W m⁻²]", cb -> cb.osr,  101.9),
        ("SRD [W m⁻²]", cb -> cb.srd,  168.0),
        ("SRU [W m⁻²]", cb -> cb.sru,   23.0),
    ]
    fig_rad = Figure(size=(1400, 320), fontsize=13)
    for (col, (title, extractor, ref)) in enumerate(sw_panels)
        ax = Axis(fig_rad[1, col]; xlabel="Day", ylabel="W m⁻²",
                  title=title, ax_kw...)
        lines!(ax, default_cb.days, extractor(default_cb); color=:gray50, label="default")
        lines!(ax, trained_cb.days, extractor(trained_cb); color=:royalblue, label="trained")
        hlines!(ax, [ref]; color=:black, linestyle=:dash, linewidth=1, label="Trenberth")
        col == 1 && axislegend(ax; position=:rt)
    end

    # LW budget
    lw_panels = [
        ("OLR [W m⁻²]", cb -> cb.olr,  235.0),
        ("LRD [W m⁻²]", cb -> cb.lrd,  333.0),
        ("LRU [W m⁻²]", cb -> cb.lru,  398.0),
    ]
    fig_lw = Figure(size=(1400, 320), fontsize=13)
    for (col, (title, extractor, ref)) in enumerate(lw_panels)
        ax = Axis(fig_lw[1, col]; xlabel="Day", ylabel="W m⁻²",
                  title=title, ax_kw...)
        lines!(ax, default_cb.days, extractor(default_cb); color=:gray50, label="default")
        lines!(ax, trained_cb.days, extractor(trained_cb); color=:tomato,    label="trained")
        hlines!(ax, [ref]; color=:black, linestyle=:dash, linewidth=1, label="Trenberth")
        col == 1 && axislegend(ax; position=:rt)
    end

    # Precipitation
    fig_precip = Figure(size=(900, 320), fontsize=13)
    ax_p = Axis(fig_precip[1,1]; xlabel="Day", ylabel="mm day⁻¹",
                title="Total precipitation", ax_kw...)
    lines!(ax_p, default_cb.days, default_cb.precip_total; color=:gray50, label="default")
    lines!(ax_p, trained_cb.days, trained_cb.precip_total; color=:royalblue, label="trained")
    hlines!(ax_p, [2.74]; color=:black, linestyle=:dash, linewidth=1, label="ERA5 ~2.74")
    axislegend(ax_p; position=:rt)

    # Trenberth summary bar chart
    summary_keys = [("OSR",  s -> s.osr,  101.9),
                    ("OLR",  s -> s.olr,  235.0),
                    ("SRD",  s -> s.srd,  168.0),
                    ("SRU",  s -> s.sru,   23.0),
                    ("LRD",  s -> s.lrd,  333.0),
                    ("LRU",  s -> s.lru,  398.0)]
    labels  = [t for (t,_,_) in summary_keys]
    refs    = Float64[r    for (_,_,r) in summary_keys]
    def_v   = Float64[f(validation.default) for (_,f,_) in summary_keys]
    train_v = Float64[f(validation.trained) for (_,f,_) in summary_keys]
    def_err   = def_v   .- refs
    train_err = train_v .- refs

    fig_summary = Figure(size=(900, 400), fontsize=13)
    ax_s = Axis(fig_summary[1,1]; xticks=(1:length(labels), labels),
                ylabel="Bias vs Trenberth [W m⁻²]",
                title="Equilibrium flux biases", ax_kw...)
    xs = 1:length(labels)
    barplot!(ax_s, xs .- 0.2, def_err;   width=0.35, color=:gray60,    label="default")
    barplot!(ax_s, xs .+ 0.2, train_err; width=0.35, color=:royalblue, label="trained")
    hlines!(ax_s, [0.0]; color=:black, linewidth=1)
    axislegend(ax_s; position=:rt)

    # Vertical temperature profile + anomaly flags
    fig_temp = SpeedyCalibration.plot_vertical_profile(
        validation.trained.temp_profile;
        reference   = validation.default.temp_profile,
        label       = "trained",
        ref_label   = "default",
        title       = "Equilibrium temperature profile",
    )

    if !isnothing(save_dir)
        mkpath(save_dir)
        save(joinpath(save_dir, "fig_clm_rad.pdf"),     fig_rad)
        save(joinpath(save_dir, "fig_clm_lw.pdf"),      fig_lw)
        save(joinpath(save_dir, "fig_clm_precip.pdf"),  fig_precip)
        save(joinpath(save_dir, "fig_clm_summary.pdf"), fig_summary)
        save(joinpath(save_dir, "fig_clm_temp.pdf"),    fig_temp)
    end

    return (; fig_rad, fig_lw, fig_precip, fig_summary, fig_temp)
end

# plot_vertical_profile

"""
    plot_vertical_profile(profile; reference=nothing, label="profile", ref_label="reference",
                           ratio_threshold=1.5, title="Vertical profile") -> Figure

Plot a vertical profile (e.g. the temperature profile from `DailyMeansCallback`)
against layer index, optionally overlaid with a `reference` profile. Layers
flagged by [`SpeedyCalibration.vertical_gradient_anomalies`](@ref) are marked
with a red ring.
"""
function SpeedyCalibration.plot_vertical_profile(
        profile::AbstractVector;
        reference::Union{Nothing,AbstractVector} = nothing,
        label::AbstractString     = "profile",
        ref_label::AbstractString = "reference",
        ratio_threshold::Real      = 1.5,
        title::AbstractString      = "Vertical profile",
    )
    layers = 1:length(profile)
    fig = Figure(size=(500, 450), fontsize=13)
    ax  = Axis(fig[1,1]; xlabel="Value", ylabel="Layer (1 = top)",
               title=title, yreversed=true,
               xgridvisible=false, ygridvisible=false,
               topspinevisible=false, rightspinevisible=false)
    lines!(ax, profile, layers; color=:royalblue, linewidth=2, label=label)
    if !isnothing(reference)
        lines!(ax, reference, layers; color=:gray50, linewidth=2, linestyle=:dash, label=ref_label)
    end

    anomalies = SpeedyCalibration.vertical_gradient_anomalies(profile; ratio_threshold)
    for a in anomalies
        scatter!(ax, [profile[a.layer+1]], [a.layer+1]; color=:red, markersize=14,
                 strokewidth=2, strokecolor=:red, marker=:circle, glowwidth=0)
    end

    axislegend(ax; position=:rb)
    return fig
end

# plot_nan_watch

"""
    plot_nan_watch(cb::NaNWatchCallback; save_dir=nothing) -> Figure

Plot non-finite counts and max |value| over time for each field tracked by a
[`NaNWatchCallback`](@ref). Generalises the per-scheme NaN timeseries plots
from `diagnose_longwave_timeseries.jl` to any tracked field set.
"""
function SpeedyCalibration.plot_nan_watch(
        cb::NaNWatchCallback;
        save_dir::Union{Nothing,AbstractString} = nothing,
    )
    ax_kw = (xgridvisible=false, ygridvisible=false,
             topspinevisible=false, rightspinevisible=false)
    fields = cb.fields
    fig = Figure(size=(900, 300*length(fields)), fontsize=13)
    for (row, f) in enumerate(fields)
        ax1 = Axis(fig[row, 1]; xlabel="Day", ylabel="count",
                    title="$(f): non-finite / blow-up (>$(cb.bound)) count", ax_kw...)
        lines!(ax1, cb.days, cb.n_nonfinite[f]; color=:tomato,    label="non-finite")
        lines!(ax1, cb.days, cb.n_blowup[f];    color=:darkorange, label="blow-up")
        row == 1 && axislegend(ax1; position=:rt)

        ax2 = Axis(fig[row, 2]; xlabel="Day", ylabel="max |value|",
                    title="$(f): max magnitude", ax_kw...)
        lines!(ax2, cb.days, cb.max_abs[f]; color=:royalblue)
    end

    if !isnothing(save_dir)
        mkpath(save_dir)
        save(joinpath(save_dir, "fig_nan_watch.pdf"), fig)
    end
    return fig
end

# Spatial map plotting

"""
    plot_field_map(lons, lats, data; colormap=:viridis, colorrange=nothing,
                    title="", coastlines=true) -> Figure

Plot a single `(lon, lat)` field as a global heatmap with optional coastlines.
Generalises `plot_netcdf_variable` (`helpers/plotting.jl`) to arbitrary
lon/lat-gridded data (model output or reference/satellite data alike).
"""
function SpeedyCalibration.plot_field_map(
        lons::AbstractVector, lats::AbstractVector, data::AbstractMatrix;
        colormap = :viridis,
        colorrange::Union{Nothing,Tuple} = nothing,
        title::AbstractString = "",
        coastlines::Bool = true,
    )
    finite = filter(isfinite, vec(data))
    clim = isnothing(colorrange) ?
        (isempty(finite) ? (0.0, 1.0) : (minimum(finite), maximum(finite))) : colorrange

    fig = Figure(size=(800, 450), fontsize=13)
    ax  = Axis(fig[1,1]; title=title, aspect=DataAspect())
    hm  = heatmap!(ax, lons, lats, data; colormap, colorrange=clim)
    coastlines && lines!(ax, GeoMakie.coastlines(); color=:black, linewidth=0.6)
    hidedecorations!(ax)
    Colorbar(fig[1,2]; colormap, limits=clim)
    return fig
end

"""
    plot_native_field(field; colormap=:viridis, colorrange=nothing,
                       coastline_color=:white, title="") -> Figure

Plot a `RingGrids` field directly on its native grid as cell polygons (no
interpolation). Generalises `plot_native!` from
`validation/chapter3_validation_plots.ipynb` into a standalone, reusable
function returning its own `Figure`.
"""
function SpeedyCalibration.plot_native_field(
        field;
        colormap = :viridis,
        colorrange::Union{Nothing,Tuple} = nothing,
        coastline_color = :white,
        title::AbstractString = "",
    )
    finite = filter(isfinite, vec(field.data))
    clim = isnothing(colorrange) ?
        (isempty(finite) ? (0.0, 1.0) : (minimum(finite), maximum(finite))) : colorrange

    fig = Figure(size=(800, 450), fontsize=13)
    ax  = Axis(fig[1,1]; title=title)
    faces = RingGrids.get_gridcell_polygons(field.grid)
    wrap(v) = (v[1] > 180 ? v[1] - 360 : v[1], v[2])
    polygons = [Polygon(Point.(wrap.(faces[:, ij]))) for ij in axes(faces, 2)]
    poly!(ax, polygons; color=field.data, colormap, colorrange=clim, strokewidth=0)
    lines!(ax, GeoMakie.coastlines(); color=coastline_color, linewidth=0.7)
    hidedecorations!(ax)
    Colorbar(fig[1,2]; colormap, limits=clim)
    return fig
end

"""
    plot_comparison_map(sim, ref, lons, lats; sim_label="Simulation",
                         ref_label="Reference", colormap=:viridis,
                         diff_colormap=:RdBu, title_prefix="") -> Figure

Three-panel comparison map: `sim`, `ref`, and their difference (`sim - ref`)
with a symmetric diverging colormap. Generalises `create_comparison_plot`
(`helpers/plotting.jl`) and the EUMETSAT/CESM2 comparison-plot functions
(`shortwave/compare_osr_monthly_eumetsat.jl`) into one function usable for
any sim-vs-reference field (OSR, temperature, precip, ...).
"""
function SpeedyCalibration.plot_comparison_map(
        sim::AbstractMatrix, ref::AbstractMatrix,
        lons::AbstractVector, lats::AbstractVector;
        sim_label::AbstractString = "Simulation",
        ref_label::AbstractString = "Reference",
        colormap = :viridis,
        diff_colormap = :RdBu,
        title_prefix::AbstractString = "",
        coastlines::Bool = true,
    )
    size(sim) == size(ref) || error("sim/ref size mismatch: $(size(sim)) vs $(size(ref))")
    diffmap = sim .- ref

    all_vals  = filter(isfinite, vcat(vec(sim), vec(ref)))
    data_clim = isempty(all_vals) ? (0.0, 1.0) : (minimum(all_vals), maximum(all_vals))
    data_clim[1] == data_clim[2] && (data_clim = (data_clim[1]-0.1, data_clim[1]+0.1))

    diff_vals = filter(isfinite, vec(diffmap))
    max_abs   = isempty(diff_vals) ? 1.0 : maximum(abs.(diff_vals))
    diff_clim = (-max_abs, max_abs)

    fig = Figure(size=(1500, 420), fontsize=13)
    for (col, (lab, data, cmap, clim)) in enumerate((
            (sim_label, sim, colormap, data_clim),
            (ref_label, ref, colormap, data_clim),
            ("Difference ($sim_label − $ref_label)", diffmap, diff_colormap, diff_clim),
        ))
        ax = Axis(fig[1, col]; title=lab, aspect=DataAspect())
        heatmap!(ax, lons, lats, data; colormap=cmap, colorrange=clim)
        coastlines && lines!(ax, GeoMakie.coastlines(); color=:black, linewidth=0.5)
        hidedecorations!(ax)
    end
    Colorbar(fig[1,4]; colormap, limits=data_clim)
    Colorbar(fig[1,5]; colormap=diff_colormap, limits=diff_clim, label="Difference")

    if !isempty(title_prefix)
        fig[0, :] = Label(fig, title_prefix, fontsize=16)
    end
    return fig
end

# plot_experiment_comparison

"""
    plot_experiment_comparison(results::Vector{TrainingResult}; labels=nothing) -> NamedTuple

Compare multiple `TrainingResult`s side by side: smoothed loss curves overlaid,
and final parameter values as a grouped bar chart (normalised to each
parameter's bounds). Generalises the cross-experiment comparison plots from
`differentiability/gpu_experiments/src/plotting.jl` to any set of
`SpeedyCalibration.calibrate!` runs.
"""
function SpeedyCalibration.plot_experiment_comparison(
        results::Vector{TrainingResult};
        labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
        save_dir::Union{Nothing,AbstractString} = nothing,
    )
    isempty(results) && error("plot_experiment_comparison: results is empty")
    labs = isnothing(labels) ? ["run $i" for i in 1:length(results)] : labels
    colors = wong_colors()

    ax_kw = (xgridvisible=false, ygridvisible=false,
             topspinevisible=false, rightspinevisible=false)

    # Loss curves overlay
    fig_loss = Figure(size=(900, 400), fontsize=13)
    ax_l = Axis(fig_loss[1,1]; xlabel="Batch", ylabel="Smoothed loss",
                title="Loss comparison across runs", yscale=log10, ax_kw...)
    for (i, r) in enumerate(results)
        lines!(ax_l, r.history[:batch], r.history[:smoothed_loss];
               color=colors[mod1(i, length(colors))], linewidth=2, label=labs[i])
    end
    axislegend(ax_l; position=:rt)

    # Parameter endpoints
    all_names = unique(vcat([[s.name for s in r.param_specs] for r in results]...))
    fig_params = Figure(size=(max(500, 200*length(all_names)), 400), fontsize=13)
    ax_p = Axis(fig_params[1,1]; xticks=(1:length(all_names), string.(all_names)),
                ylabel="Normalised value (0=lower, 1=upper bound)",
                title="Final parameter values", ax_kw...)
    nres = length(results)
    width = 0.8 / nres
    for (i, (r, lab)) in enumerate(zip(results, labs))
        xs = Float64[]
        ys = Float64[]
        for (j, name) in enumerate(all_names)
            spec = findfirst(s -> s.name == name, r.param_specs)
            isnothing(spec) && continue
            s = r.param_specs[spec]
            lo, hi = s.bounds
            v = get(r.final_params, name, NaN32)
            push!(xs, j + (i - (nres+1)/2) * width)
            push!(ys, (v - lo) / (hi - lo))
        end
        barplot!(ax_p, xs, ys; width=width*0.9, color=colors[mod1(i, length(colors))], label=lab)
    end
    ylims!(ax_p, -0.05, 1.05)
    axislegend(ax_p; position=:rt)

    if !isnothing(save_dir)
        mkpath(save_dir)
        save(joinpath(save_dir, "fig_compare_loss.pdf"),   fig_loss)
        save(joinpath(save_dir, "fig_compare_params.pdf"), fig_params)
    end

    return (; fig_loss, fig_params)
end

end # module
