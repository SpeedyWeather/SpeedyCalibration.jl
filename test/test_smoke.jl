# Integration smoke test: 5-batch calibration at trunc=5.
# Verifies the full calibrate! pipeline runs without error and history is populated.

@testset "calibrate! smoke test" begin
    params = [
        ParamSpec(:cloud_albedo, [:shortwave_radiation, :clouds, :cloud_albedo];
                  bounds=(0.1f0, 0.85f0), initial=0.4f0),
    ]

    cfg = quick_test_config()
    result = calibrate!(params, Optimisers.Adam(1f-2), OSR_LOSS, cfg)

    @test result isa TrainingResult
    @test length(result.history[:batch]) == cfg.max_batches ||
          result.conv_info.converged
    @test isfinite(result.conv_info.best_smoothed_loss)
    @test haskey(result.final_params, :cloud_albedo)
    val = result.final_params[:cloud_albedo]
    @test val >= 0.1f0
    @test val <= 0.85f0
end
