# Installation

SpeedyCalibration.jl is not yet registered in the Julia General Registry.
Install it directly from GitHub:

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/SpeedyWeather/SpeedyCalibration.jl")
```

or in the package manager (`]` opens it):

```julia
(@v1.10) pkg> add https://github.com/SpeedyWeather/SpeedyCalibration.jl
```

If you are working on the thesis repository directly, the package lives at
`Code_SpeedyWeather/SpeedyCalibration.jl/`. Activate it with

```julia
julia> Pkg.develop(path="path/to/SpeedyCalibration.jl")
```

## Dependencies

SpeedyCalibration.jl requires:

- Julia 1.10 or later
- [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) ≥ 0.21
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) ≥ 0.13
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) ≥ 0.3

These are all listed in `Project.toml` and installed automatically.

## Plotting extension

Visualisation functions (`plot_training`, `plot_climate`) are available as a
[weak dependency](https://pkgdocs.julialang.org/v1.10/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions))
on CairoMakie + GeoMakie. The package loads without them; install and load both separately
when you need figures:

```julia
(@v1.10) pkg> add CairoMakie GeoMakie
```

```julia
using SpeedyCalibration
using CairoMakie, GeoMakie
```

!!! note "Julia version"
    Enzyme.jl currently works best on Julia 1.10. If you run into AD compilation
    issues on Julia 1.11+, switch to 1.10. SpeedyWeather.jl's own
    [differentiability docs](https://speedyweather.github.io/SpeedyWeather.jl/dev/differentiability/)
    track the current status.
