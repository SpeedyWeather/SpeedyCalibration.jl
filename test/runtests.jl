using Test
using SpeedyCalibration
using SpeedyWeather
using Optimisers
using Dates

@testset "SpeedyCalibration" begin
    include("test_param_spec.jl")
    include("test_diagnostics.jl")
    "--smoke" in ARGS && include("test_smoke.jl")
end
