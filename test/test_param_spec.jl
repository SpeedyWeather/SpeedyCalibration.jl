@testset "ParamSpec" begin
    spec = ParamSpec(:cloud_albedo, [:shortwave_radiation, :clouds, :cloud_albedo];
                     bounds=(0.1f0, 0.9f0), initial=0.5f0, grad_scale=2.0f0)
    @test spec.name == :cloud_albedo
    @test spec.bounds == (0.1f0, 0.9f0)
    @test spec.initial == 0.5f0
    @test spec.grad_scale == 2.0f0
end

@testset "Sigmoid reparameterisation" begin
    lb, ub = 0.1f0, 0.9f0

    # round-trip: physical → raw → physical
    for θ in [0.2f0, 0.5f0, 0.8f0]
        raw = SpeedyCalibration.to_raw(θ, lb, ub)
        @test SpeedyCalibration.sigmoid_param(raw, lb, ub) ≈ θ atol=1f-5
    end

    # bounds are respected
    raw_lb = SpeedyCalibration.to_raw(lb + 1f-3, lb, ub)
    raw_ub = SpeedyCalibration.to_raw(ub - 1f-3, lb, ub)
    @test SpeedyCalibration.sigmoid_param(raw_lb - 100f0, lb, ub) >= lb
    @test SpeedyCalibration.sigmoid_param(raw_ub + 100f0, lb, ub) <= ub

    # grad factor is positive and shrinks near bounds
    mid = SpeedyCalibration.to_raw(0.5f0, lb, ub)
    near_lb = SpeedyCalibration.to_raw(lb + 0.01f0, lb, ub)
    gf_mid  = SpeedyCalibration.sigmoid_grad_factor(mid, lb, ub)
    gf_edge = SpeedyCalibration.sigmoid_grad_factor(near_lb, lb, ub)
    @test gf_mid > 0
    @test gf_edge < gf_mid
end

@testset "get_by_path / set_by_path!" begin
    mutable struct _Inner; x::Float32; end
    mutable struct _Outer; inner::_Inner; end
    obj = _Outer(_Inner(1.0f0))
    path = [:inner, :x]
    @test SpeedyCalibration.get_by_path(obj, path) == 1.0f0
    SpeedyCalibration.set_by_path!(obj, path, 2.0f0)
    @test SpeedyCalibration.get_by_path(obj, path) == 2.0f0
end

@testset "LossConfig" begin
    cfg = OSR_SRU_SRD_LOSS
    @test :osr in cfg.flux_keys
    @test cfg.targets[:osr] ≈ 101.9f0

    # unknown flux key
    @test_throws ErrorException LossConfig([:xyz];
        targets=Dict(:xyz => 1f0), weights=Dict(:xyz => 1f0))

    # missing target
    @test_throws ErrorException LossConfig([:osr];
        targets=Dict(:sru => 1f0), weights=Dict(:osr => 1f0))
end

@testset "compute_loss" begin
    cfg = OSR_LOSS
    perfect = Dict(:osr => 101.9f0)
    @test SpeedyCalibration.compute_loss(perfect, cfg) ≈ 0f0 atol=1f-5

    off = Dict(:osr => 110.0f0)
    expected = (110f0 - 101.9f0)^2
    @test SpeedyCalibration.compute_loss(off, cfg) ≈ expected atol=1f-3

    # NaN propagates to Inf
    nan = Dict(:osr => NaN32)
    @test isinf(SpeedyCalibration.compute_loss(nan, cfg))
end

@testset "TrainingConfig" begin
    cfg = TrainingConfig()
    @test cfg.spinup_days == 180
    @test cfg.max_batches == 300

    qcfg = quick_test_config()
    @test qcfg.trunc == 5
    @test qcfg.max_batches == 5
end
