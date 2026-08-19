"""
    enzyme_warmup(sim, loss_config, param_specs)

Force Enzyme to compile its AD rules by running one backward pass on `sim`,
the **same** model that will be used for training. Must be called after the
spinup so that radiation and parameterization fields are physically initialised;
running AD on a freshly-initialised (all-zero flux) state produces NaN gradients.

Called automatically by `calibrate!` after spinup when `TrainingConfig.warmup_enzyme = true`
(the default). You rarely need to call this directly.
"""
function enzyme_warmup(sim, loss_config::LossConfig, param_specs::Vector{ParamSpec})
    println("Warming up Enzyme (compiling AD rules on actual model)...")
    flush(stdout)
    t0 = time()
    compute_gradients!(sim.variables, sim.model, loss_config, param_specs)
    @printf("Enzyme warmup complete in %.1f s.\n", time() - t0)
    flush(stdout)
    return nothing
end

"""
    compute_gradients!(variables, model, loss_config, param_specs)
        → (gradients, means, loss)

Single-timestep reverse-mode AD pass (Enzyme). Returns:
- `gradients::Vector{Float32}`: `∂L/∂θ` for each `ParamSpec`
- `means::Dict{Symbol,Float32}`: area-weighted global mean of each tracked flux
- `loss::Float32`: current loss value

The function saves the full prognostic state before the Enzyme call and restores it
afterwards so the simulation continues from the same point.
Returns `(zeros, NaN-means, Inf)` if any flux mean is non-finite.
"""
function compute_gradients!(
        variables,
        model,
        cfg::LossConfig,
        param_specs::Vector{ParamSpec},
    )
    progn = variables.prognostic

    # Area-weighted means
    means, wsums = compute_flux_means(variables, cfg.flux_keys)
    if any(!isfinite(means[k]) for k in cfg.flux_keys)
        nan_means = Dict{Symbol,Float32}(k => NaN32 for k in cfg.flux_keys)
        return fill(0f0, length(param_specs)), nan_means, Inf32
    end

    loss   = compute_loss(means, cfg)
    coeffs = loss_coefficients(means, cfg)

    # Save full prognostic state
    saved_vor   = deepcopy(progn.vorticity)
    saved_div   = deepcopy(progn.divergence)
    saved_temp  = deepcopy(progn.temperature)
    saved_hum   = deepcopy(progn.humidity)
    saved_pres  = deepcopy(progn.pressure)
    saved_ocean = deepcopy(progn.ocean)
    saved_land  = deepcopy(progn.land)
    saved_grid  = deepcopy(variables.grid)

    model.feedback.nans_detected = false

    dvariables = Enzyme.make_zero(variables)
    dmodel     = Enzyme.make_zero(model)

    # Seed cotangents per ring
    # seed_ij = coeff_k · gw_j / (nlons_j · w_sum_k)  (chain rule through area mean)
    param = variables.parameterizations
    first_field = getproperty(param, FLUX_FIELD[cfg.flux_keys[1]])
    _grid  = first_field.grid
    _gw    = Float32.(RingGrids.gaussian_weights(_grid.nlat_half))
    _rings = eachring(_grid)

    dparam = dvariables.parameterizations
    for (j, ring) in enumerate(_rings)
        nlons_j = length(ring)
        seeds = Dict{Symbol,Float32}(
            k => coeffs[k] * _gw[j] / (Float32(nlons_j) * wsums[k]) for k in cfg.flux_keys
        )
        for i in ring, k in cfg.flux_keys
            field  = getproperty(param,  FLUX_FIELD[k])
            dfield = getproperty(dparam, FLUX_FIELD[k])
            dfield[i] = isfinite(field[i]) ? seeds[k] : 0f0
        end
    end

    # Enzyme reverse pass. `time_step!(vars, time_stepping, model)` is the dynamics+physics
    # step only (no clock advance) — matches the old `timestep!(variables, dt, model)`
    # semantics this call previously relied on before SpeedyWeather.jl replaced `timestep!`
    # with the `time_step!` family (SpeedyWeather 0.22).
    Enzyme.autodiff(Enzyme.Reverse, SpeedyWeather.time_step!, Const,
        Duplicated(variables, dvariables),
        Const(model.time_stepping),
        Duplicated(model, dmodel))

    model.feedback.nans_detected = false

    # Restore prognostic state
    progn.vorticity   .= saved_vor
    progn.divergence  .= saved_div
    progn.temperature .= saved_temp
    progn.humidity    .= saved_hum
    progn.pressure    .= saved_pres
    for key in keys(progn.ocean); progn.ocean[key] .= saved_ocean[key]; end
    for key in keys(progn.land);  progn.land[key]  .= saved_land[key];  end
    for fn in fieldnames(typeof(variables.grid))
        dst = getfield(variables.grid, fn)
        src = getfield(saved_grid, fn)
        dst isa AbstractArray && copyto!(dst, src)
    end

    gradients = Float32[Float32(get_by_path(dmodel, spec.path)) for spec in param_specs]
    return gradients, means, loss
end
