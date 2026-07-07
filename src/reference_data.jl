"""
    load_netcdf_field(filepath, varname; verbose=false) -> (data, lons, lats)

Read a 1D/2D/3D variable from a NetCDF file and return it as a `(lon, lat)` or
`(lon, lat, time)` `Float64` array (missing → NaN), together with the
`lon`/`lat` coordinate vectors. Auto-reshapes flattened 1D fields using the
file's `lon`/`lat`(/`time`) dimensions.

Generalises the file-reading logic duplicated across `plot_netcdf_variable`
(`helpers/plotting.jl`) and the EUMETSAT/CESM2 comparison scripts.
"""
function load_netcdf_field(filepath::AbstractString, varname::AbstractString; verbose::Bool=false)
    isfile(filepath) || error("NetCDF file not found: $filepath")
    return NCDatasets.NCDataset(filepath, "r") do ds
        haskey(ds, varname) || error(
            "Variable \"$varname\" not found in $filepath. Available: $(collect(keys(ds)))")
        data = ds[varname][:]
        lons = Float64.(ds["lon"][:])
        lats = Float64.(ds["lat"][:])

        if ndims(data) == 1
            nlon, nlat = length(lons), length(lats)
            if haskey(ds, "time") && length(data) == nlon * nlat * length(ds["time"])
                data = reshape(data, nlon, nlat, length(ds["time"]))
            elseif length(data) == nlon * nlat
                data = reshape(data, nlon, nlat)
            else
                error("Cannot reshape 1D variable \"$varname\" (length $(length(data))) " *
                      "to grid $nlon × $nlat")
            end
        end

        clean = Array{Float64}(undef, size(data))
        for i in eachindex(data)
            clean[i] = ismissing(data[i]) ? NaN : Float64(data[i])
        end

        if verbose
            finite = filter(isfinite, vec(clean))
            println("$varname: shape=$(size(clean))  " *
                    "min=$(round(minimum(finite), digits=2))  " *
                    "max=$(round(maximum(finite), digits=2))  " *
                    "mean=$(round(mean(finite), digits=2))")
        end
        return clean, lons, lats
    end
end

"""
    time_mean(data; weights=nothing) -> Matrix

Average a `(lon, lat, time)` array over its last (time) dimension. Pass
`weights` (a vector the length of the time dimension) for a time-weighted mean.
No-op (returns `data` unchanged) for already-2D input.
"""
function time_mean(data::AbstractArray; weights::Union{Nothing,AbstractVector}=nothing)
    ndims(data) == 2 && return data
    ndims(data) == 3 || error("time_mean expects a 2D or 3D array, got ndims=$(ndims(data))")
    nx, ny, nt = size(data)
    if weights === nothing
        return dropdims(mean(data, dims=3), dims=3)
    end
    length(weights) == nt || error("weights length $(length(weights)) ≠ time dimension $nt")
    out = zeros(eltype(data), nx, ny)
    wsum = sum(weights)
    for i in 1:nx, j in 1:ny
        out[i, j] = sum(data[i, j, :] .* weights) / wsum
    end
    return out
end

"""
    regrid_nearest(data, src_lons, src_lats, dst_lons, dst_lats) -> Matrix

Nearest-neighbour regrid of `data` (defined on `src_lons`/`src_lats`) onto the
`dst_lons`/`dst_lats` grid, handling the antimeridian wraparound
(0°/360° vs. −180°/180° conventions). Generalises
`interpolate_to_common_grid_with_coords` from `compare_osr_monthly_eumetsat.jl`
to arbitrary source/destination grids.
"""
function regrid_nearest(data::AbstractMatrix,
                         src_lons::AbstractVector, src_lats::AbstractVector,
                         dst_lons::AbstractVector, dst_lats::AbstractVector)
    lon_idx = Vector{Int}(undef, length(dst_lons))
    for (i, lon) in enumerate(dst_lons)
        d = min.(abs.(src_lons .- lon), abs.(src_lons .- lon .+ 360), abs.(src_lons .- lon .- 360))
        lon_idx[i] = argmin(d)
    end
    lat_idx = Vector{Int}(undef, length(dst_lats))
    for (j, lat) in enumerate(dst_lats)
        lat_idx[j] = argmin(abs.(src_lats .- lat))
    end
    out = Matrix{Float64}(undef, length(dst_lons), length(dst_lats))
    for i in eachindex(dst_lons), j in eachindex(dst_lats)
        out[i, j] = data[lon_idx[i], lat_idx[j]]
    end
    return out
end

"""
    compare_to_reference(sim, ref) -> NamedTuple

Pointwise comparison statistics between a simulated field and a reference
field on the same grid: mean bias, RMSE, max |diff|, and the Pearson
correlation. NaNs are excluded pairwise. Generalises
`print_comparison_stats` from `compare_osr_monthly_eumetsat.jl`.
"""
function compare_to_reference(sim::AbstractArray, ref::AbstractArray)
    size(sim) == size(ref) || error("sim/ref size mismatch: $(size(sim)) vs $(size(ref))")
    mask = isfinite.(sim) .& isfinite.(ref)
    s, r = sim[mask], ref[mask]
    isempty(s) && error("no finite, overlapping points between sim and ref")
    diff = s .- r
    return (
        n             = length(s),
        mean_sim      = mean(s),
        mean_ref      = mean(r),
        mean_bias     = mean(diff),
        rmse          = sqrt(mean(diff .^ 2)),
        max_abs_diff  = maximum(abs.(diff)),
        correlation   = cor(s, r),
    )
end
