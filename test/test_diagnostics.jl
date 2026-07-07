@testset "check_nans" begin
    clean = [1.0, 2.0, 3.0]
    r = check_nans(clean; label="clean")
    @test r.ok
    @test r.n_nan == 0
    @test r.n_inf == 0
    @test r.max_finite == 3.0

    dirty = [1.0, NaN, Inf, -Inf, 2.0]
    r2 = check_nans(dirty; label="dirty")
    @test !r2.ok
    @test r2.n_nan == 1
    @test r2.n_inf == 2
    @test r2.frac_bad ≈ 3/5
    @test r2.max_finite == 2.0
end

@testset "vertical_gradient_anomalies" begin
    # smooth profile: no anomalies
    smooth = collect(1.0:8.0)
    @test isempty(vertical_gradient_anomalies(smooth))

    # layer 1→2 jump much larger than layer 2→3
    spike = [1.0, 20.0, 21.0, 22.0, 23.0]
    flags = vertical_gradient_anomalies(spike; ratio_threshold=1.5)
    @test length(flags) == 1
    @test flags[1].layer == 1
    @test flags[1].ratio > 1.5

    # too short to evaluate
    @test isempty(vertical_gradient_anomalies([1.0, 2.0]))
end

@testset "reference_data: regrid_nearest" begin
    src_lons = [0.0, 90.0, 180.0, 270.0]
    src_lats = [-45.0, 45.0]
    data = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]  # (lon, lat)

    # identical grid → identity regrid
    out = regrid_nearest(data, src_lons, src_lats, src_lons, src_lats)
    @test out == data

    # wraparound: target lon 359 should map to src lon 0
    out2 = regrid_nearest(data, src_lons, src_lats, [359.0], [-45.0])
    @test out2[1,1] == data[1,1]
end

@testset "reference_data: compare_to_reference" begin
    sim = [1.0 2.0; 3.0 4.0]
    ref = [1.0 2.0; 3.0 5.0]
    stats = compare_to_reference(sim, ref)
    @test stats.n == 4
    @test stats.mean_bias ≈ -0.25
    @test stats.max_abs_diff ≈ 1.0
    @test stats.correlation <= 1.0

    # NaNs excluded pairwise
    sim_nan = [1.0 NaN; 3.0 4.0]
    stats2 = compare_to_reference(sim_nan, ref)
    @test stats2.n == 3
end

@testset "reference_data: time_mean" begin
    data2d = [1.0 2.0; 3.0 4.0]
    @test time_mean(data2d) === data2d

    data3d = cat([1.0 2.0; 3.0 4.0], [3.0 4.0; 5.0 6.0]; dims=3)
    m = time_mean(data3d)
    @test m == [2.0 3.0; 4.0 5.0]

    mw = time_mean(data3d; weights=[1.0, 3.0])
    @test mw[1,1] ≈ (1.0*1.0 + 3.0*3.0) / 4.0
end

@testset "NaNWatchCallback smoke test" begin
    cfg  = quick_test_config()
    sg   = SpectralGrid(trunc=cfg.trunc, nlayers=cfg.nlayers)
    planet = Earth(sg; daily_cycle=false, seasonal_cycle=false)
    model  = PrimitiveWetModel(sg; planet=planet)
    cb     = NaNWatchCallback(schedule=Schedule(every=Hour(6)))
    add!(model, :nan_watch => cb)
    sim = initialize!(model)
    run!(sim, period=Day(1), output=false)

    @test length(cb.days) > 0
    @test all(isfinite, cb.days)
    for f in cb.fields
        @test length(cb.n_nonfinite[f]) == length(cb.days)
    end
    @test any_anomaly(cb) isa Bool
end
