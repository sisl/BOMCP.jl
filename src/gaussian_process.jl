mutable struct GPLA{BD, AD, X<:AbstractMatrix, Y<:AbstractVector, M<:GaussianProcesses.Mean, K<:GaussianProcesses.Kernel, NOI<:GaussianProcesses.Param} <: GaussianProcesses.GPBase
    x::X
    y::Y
    k::Int64 #Local Neighborhood Size

    #=
    action_dims::Int64
    belief_dims::Int64
    dim::Int64
    =#

    mean::M
    kernel::K
    logNoise::NOI

    kdtree::KDTree

    "Auxiliary variables used to optimize GP hyperparameters"
    data::GaussianProcesses.KernelData
    mll::Float64
    dmll::Vector{Float64}
    target::Float64
    dtarget::Vector{Float64}
    function GPLA{BD, AD, X, Y, M, K, NOI}(x::X, y::Y, k::Int64, mean::M, kernel::K, logNoise::NOI) where {BD, AD, X, Y, N, M, K, NOI}
        data = GaussianProcesses.KernelData(kernel, x, x)
        tree = KDTree(zeros(Float64, AD + BD, 0), Euclidean())
        gp = new{BD, AD, X, Y, M, K, NOI}(ElasticArray(x), ElasticArray(y), k, mean, kernel, logNoise, tree, data)
        initialize!(gp)
    end
end

function GPLA(x::AbstractMatrix, y::AbstractVector, k::Integer, action_dims::Integer, belief_dims::Integer, mean::GaussianProcesses.Mean, kernel::GaussianProcesses.Kernel, logNoise::Real)
    lns = GaussianProcesses.wrap_param(logNoise)
    size_x = size(x, 2)
    GPLA{belief_dims, action_dims, typeof(x),typeof(y),typeof(mean),typeof(kernel), typeof(lns)}(x, y, k, mean, kernel, lns)
end

beliefdim(::GPLA{BD}) where BD = BD
actiondim(::GPLA{BD, AD}) where {BD, AD} = AD

function initialize!(gp::GPLA)
    n_obs = size(gp.y, 1)
    if n_obs != 0
        gp.kdtree = KDTree(gp.x)
        update_mll!(gp)
    end
    return gp
end

function GaussianProcesses.predict_f(gp::GPLA, x::AbstractArray{T,2} where T)
    nx = size(gp.x, 2)
    if nx <= gp.k
        mx = GaussianProcesses.mean(gp.mean, gp.x)
        mf = GaussianProcesses.mean(gp.mean, x)
        Kxf = GaussianProcesses.cov(gp.kernel, x, gp.x) #size(size(x,2) x nx)
        Kff = GaussianProcesses.cov(gp.kernel, x, x) .+ exp(2*gp.logNoise.value) .+ eps()
        y = gp.y - mx
        data = GaussianProcesses.KernelData(gp.kernel, gp.x, gp.x)
        Σ = GaussianProcesses.cov(gp.kernel, gp.x, gp.x, data) + Matrix(I, nx, nx).*(exp(2*gp.logNoise.value)+eps())
        cK = PDMat(GaussianProcesses.make_posdef!(Σ)...)
        α = reshape(cK \ y, nx, 1)
        β = cK \ transpose(Kxf)
        μ = mf + Kxf*α
        σ² = diag(Kff - Kxf*β)
    else
        sx = NearestNeighbors.copy_svec(Float64, extract_value(x), Val(beliefdim(gp) + actiondim(gp)))
        neighbors, _ = knn(gp.kdtree, sx, gp.k, true)
        μ = zeros(eltype(x), size(x, 2), 1)
        σ² = zeros(eltype(x), size(x, 2), 1)
        for i = 1:size(x,2)
            x_obs = gp.x[:, neighbors[i]]
            y_obs = gp.y[neighbors[i]]
            m, s = predict_local(x[:,i:i], x_obs, y_obs, gp.mean, gp.kernel, gp.logNoise)
            μ[i, 1] = m
            σ²[i, 1] = s
        end
    end
    return μ, σ²
end


function predict_local(x, x_obs, y_obs, mean, kernel, logNoise)
    k = size(x_obs, 2)
    sample_mean = Statistics.mean(y_obs)
    mx = zero(y_obs) .+ Statistics.mean(y_obs)
    mf = sample_mean
    Kxf = GaussianProcesses.cov(kernel, x, x_obs)
    Kff = GaussianProcesses.cov(kernel, x, x) .+ exp(2*logNoise.value) .+ eps()

    y_obs = reshape(y_obs - mx, size(y_obs, 1), 1)
    data = GaussianProcesses.KernelData(kernel, x_obs, x_obs)
    Σ = GaussianProcesses.cov(kernel, x_obs, x_obs, data) + Matrix(I, k, k).*(exp(2*logNoise.value)+eps())
    Kxx = PDMat(GaussianProcesses.make_posdef!(Σ)...)
    μ = mf + GaussianProcesses.dot(Kxf, Kxx \ y_obs)
    Σ = Kff - Kxf*(Kxx \ transpose(Kxf))
    σ² = max(Σ[1], 0.0)
    return μ, σ²
end

function mll_local(idx, gp, mx, neighbors)
    if idx in neighbors
        neighbors = neighbors[neighbors .!= idx]
    else
        neighbors = neighbors[1:end-1]
    end
    k = length(neighbors)

    x = gp.x[:,idx:idx]
    x_obs = gp.x[:, neighbors]
    y_obs = gp.y[neighbors] - mx[neighbors]
    y_obs = reshape(y_obs, size(y_obs, 1), 1)

    mf = mx[idx]
    Kxf = GaussianProcesses.cov(gp.kernel, x, x_obs)
    Kff = GaussianProcesses.cov(gp.kernel, x, x) .+ exp(2*gp.logNoise.value) .+ eps()

    data = GaussianProcesses.KernelData(gp.kernel, x_obs, x_obs)
    Σ = GaussianProcesses.cov(gp.kernel, x_obs, x_obs, data) + Matrix(I, gp.k, gp.k).*(exp(2*gp.logNoise.value)+eps())
    Kxx = PDMat(GaussianProcesses.make_posdef!(Σ)...)
    μ = mf + GaussianProcesses.dot(Kxf, Kxx \ y_obs)
    Σ = Kff - Kxf*(Kxx \ transpose(Kxf))
    σ² = max(Σ[1], 0.0)
    σ = sqrt(σ²)
    log_p = -0.5*((gp.y[idx] - μ)/σ)^2 - 0.5*log(2*pi) - log(σ)
    param_tuple = (μ = μ, σ²  = σ², Kxx = Kxx, Kxf = Kxf, Kff = Kff, mf = mf, y = y_obs, neighbors = neighbors)
    return log_p, param_tuple
end

function update_mll!(gp::GPLA)
    nx = size(gp.x, 2)
    μ = GaussianProcesses.mean(gp.mean, gp.x)
    if nx <= gp.k
        y = gp.y - μ
        data = GaussianProcesses.KernelData(gp.kernel, gp.x, gp.x)
        Σ = GaussianProcesses.cov(gp.kernel, gp.x, gp.x, data) + Matrix(I, nx, nx).*(exp(2*gp.logNoise.value)+eps())
        cK = PDMat(GaussianProcesses.make_posdef!(Σ)...)
        α = cK \ y
        gp.mll = - (GaussianProcesses.dot(y, α) + GaussianProcesses.logdet(cK) + log(2*pi) * nx) / 2
    else
        mx = GaussianProcesses.mean(gp.mean, gp.x)
        neighbors, _ = knn(gp.kdtree, extract_value(gp.x), gp.k + 1)
        gp.mll = 0.0
        for i = 1:nx
            log_p, _ = mll_local(i, gp, mx, neighbors[i])
            gp.mll += log_p
        end
    end
end

"""
     update_dmll!(gp::GPE, ...)
Update the gradient of the marginal log-likelihood of Gaussian process `gp`.
"""
function update_dmll!(gp::GPLA;
                    noise::Bool=true, # include gradient component for the logNoise term
                    domean::Bool=true, # include gradient components for the mean parameters
                    kern::Bool=true, # include gradient components for the spatial kernel parameters
                    )
    n_mean_params = GaussianProcesses.num_params(gp.mean)
    n_kern_params = GaussianProcesses.num_params(gp.kernel)
    gp.dmll = zeros(Float64, noise + domean * n_mean_params + kern * n_kern_params)
    nobs = size(gp.y, 1)
    μ = GaussianProcesses.mean(gp.mean, gp.x)
    neighbors, _ = knn(gp.kdtree, gp.x, gp.k + 1)
    for i = 1:nobs
        _, params = mll_local(i, gp, μ, neighbors[i])
        d_dmll = zeros(Float64, noise + domean * n_mean_params + kern * n_kern_params)
        dmll_local!(d_dmll, gp, i, n_mean_params, n_kern_params, params; noise = noise, domean = domean, kern = kern)
        gp.dmll += d_dmll
    end
end

function dmll_local!(dmll::AbstractVector, gp::GPLA, idx::Int64, n_mean_params, n_kern_params, params;
                noise::Bool=true, domean::Bool=true, kern::Bool=true)
    i=1
    if noise
        @assert GaussianProcesses.num_params(gp.logNoise) == 1
        dmll[i] = dmll_noise(gp, idx, params)
        i += 1
    end
    if domean && n_mean_params>0
        dmll_m = @view(dmll[i:i+n_mean_params-1])
        dmll_mean!(dmll_m, gp, idx, params)
        i += n_mean_params
    end
    if kern
        dmll_k = @view(dmll[i:end])
        dmll_kern!(dmll_k, gp, idx, params)
    end
    return dmll
end

function dmll_noise(gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    Knoise = diagm(repeat([2*exp(2*gp.logNoise.value)], gp.k))
    σ = sqrt(params.σ²) + eps()
    dμ = -dot(params.Kxf, params.Kxx \ Knoise * (params.Kxx \ params.y))
    dσ = dot(params.Kxf[1,:], params.Kxx \ Knoise * (params.Kxx \ params.Kxf[1,:])) + 2*exp(2*gp.logNoise.value)
    dσ /= σ*2.0
    dlog_p = -(y - params.μ)/σ*(-dμ/σ - (y - params.μ)/params.σ²*dσ)
    dlog_p -= dσ/σ
    return dlog_p
end

function dmll_mean!(dmll::AbstractVector, gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    x = gp.x[:,idx:idx]
    d_mean = -ones(Float64, gp.k)
    σ = sqrt(params.σ²) + eps()
    dμ = GaussianProcesses.grad_stack(gp.mean, x)*(1.0 + GaussianProcesses.dot(params.Kxf, params.Kxx \ d_mean))
    dσ = [0.0]
    dlog_p = -(y - params.μ)/σ*(-dμ/σ - (y - params.μ)/params.σ²*dσ)
    dlog_p -= dσ/σ

    for i in 1:length(dlog_p)
        dmll[i] = dlog_p[i]
    end
    return dmll
end

function dmll_kern!(dmll::AbstractVector, gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    x = gp.x[:,idx:idx]
    dKxx = grad_stack(gp.kernel, gp.x[:,params.neighbors], gp.x[:,params.neighbors])
    dKff = grad_stack(gp.kernel, x, x)[1, 1, :]
    dKxf = grad_stack(gp.kernel, x, gp.x[:,params.neighbors])[1,:,:]
    σ = sqrt(params.σ²) + eps()
    for i in 1:size(dKxf,2)
        dμ = GaussianProcesses.dot(dKxf[:,i], params.Kxx \ params.y)
        dμ -= GaussianProcesses.dot(params.Kxf, (params.Kxx \ dKxx[:,:,i]) * (params.Kxx \ params.y))
        dσ = dKff[i] - GaussianProcesses.dot(dKxf[:,i], params.Kxx \ params.Kxf[1,:])
        dσ += GaussianProcesses.dot(params.Kxf[1,:], (params.Kxx \ dKxx[:,:,i]) * (params.Kxx \ params.Kxf[1,:]))
        dσ -= GaussianProcesses.dot(params.Kxf[1,:], params.Kxx \ dKxf[:,i])
        dσ /= σ*2.0
        dlog_p = -(y - params.μ)/σ*(-dμ/σ - (y - params.μ)/params.σ²*dσ)
        dlog_p -= dσ/σ
        dmll[i] = dlog_p
    end
    return dmll
end

function update_mll_and_dmll!(gp::GPLA, precomp; kwargs...)
    update_mll!(gp)
    update_dmll!(gp; kwargs...)
end

function GaussianProcesses.update_target_and_dtarget!(gp::GPLA, precomp; params_kwargs...)
    update_mll_and_dmll!(gp, precomp; params_kwargs...)
    gp.target = gp.mll
    gp.dtarget = gp.dmll
end

function grad_stack(k::GaussianProcesses.Kernel, X1::AbstractMatrix, X2::AbstractMatrix)
    data = GaussianProcesses.KernelData(k, X1, X2)
    nobs1 = size(X1, 2)
    nobs2 = size(X2, 2)
    stack = Array{eltype(X1)}(undef, nobs1, nobs2, GaussianProcesses.num_params(k))
    GaussianProcesses.grad_stack!(stack, k, X1, X2, data)
end

function extract_value(x::Array)
    x_out = zeros(Float64, size(x, 1), size(x, 2))
    for j = 1:size(x,2)
        for i = 1:size(x,1)
            if x[i, j] isa AbstractFloat
                x_out[i, j] = convert(Float64, x[i, j])
            else
                x_out[i, j] = convert(Float64, x[i, j].value)
            end
        end
    end
    return x_out
end

GaussianProcesses.get_params_kwargs(::GPLA; kwargs...) = delete!(Dict(kwargs), :lik)

function GaussianProcesses.get_params(gp::GPLA; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; append!(params, GaussianProcesses.get_params(gp.logNoise)); end
    if domean
        append!(params, GaussianProcesses.get_params(gp.mean))
    end
    if kern
        append!(params, GaussianProcesses.get_params(gp.kernel))
    end
    return params
end

function GaussianProcesses.optimize!(gp::GPLA; method = GaussianProcesses.LBFGS(), domean::Bool = true, kern::Bool = true,
                   noise::Bool = true, lik::Bool = true,
                   meanbounds = nothing, kernbounds = nothing,
                   noisebounds = nothing, likbounds = nothing, kwargs...)
    params_kwargs = GaussianProcesses.get_params_kwargs(gp; domean=domean, kern=kern, noise=noise, lik=lik)
    func = GaussianProcesses.get_optim_target(gp; params_kwargs...)
    init = GaussianProcesses.get_params(gp; params_kwargs...)  # Initial hyperparameter values
    # try
        if meanbounds == kernbounds == noisebounds == likbounds == nothing
            results = Optim.optimize(func, init; method=method, kwargs...)
        else
            lb, ub = GaussianProcesses.bounds(gp, noisebounds, meanbounds, kernbounds, likbounds;
                            domean = domean, kern = kern, noise = noise, lik = lik)
            results = GaussianProcesses.optimize(func.f, func.df, lb, ub, init, Fminbox(method))
        end
        GaussianProcesses.set_params!(gp, Optim.minimizer(results); params_kwargs...)
        return results
end

function GaussianProcesses.set_params!(gp::GPLA, hyp::AbstractVector;
                     noise::Bool=true, domean::Bool=true, kern::Bool=true)
    n_noise_params = GaussianProcesses.num_params(gp.logNoise)
    n_mean_params = GaussianProcesses.num_params(gp.mean)
    n_kern_params = GaussianProcesses.num_params(gp.kernel)

    i = 1
    if noise
        GaussianProcesses.set_params!(gp.logNoise, hyp[1:n_noise_params])
        i += n_noise_params
    end

    if domean && n_mean_params>0
        GaussianProcesses.set_params!(gp.mean, hyp[i:i+n_mean_params-1])
        i += n_mean_params
    end

    if kern
        GaussianProcesses.set_params!(gp.kernel, hyp[i:i+n_kern_params-1])
        i += n_kern_params
    end
end

function GaussianProcesses.get_optim_target(gp::GPLA; params_kwargs...)
    function ltarget(hyp::AbstractVector)
        prev = GaussianProcesses.get_params(gp; params_kwargs...)
        try
            GaussianProcesses.set_params!(gp, hyp; params_kwargs...)
            update_mll!(gp)
            return -gp.mll
        catch err
            # reset parameters to remove any NaNs
            GaussianProcesses.set_params!(gp, prev; params_kwargs...)

            if !all(isfinite.(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, LinearAlgebra.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end
    end
    function ltarget_and_dltarget!(grad::AbstractVector, hyp::AbstractVector)
        prev = GaussianProcesses.get_params(gp; params_kwargs...)
        try
            GaussianProcesses.set_params!(gp, hyp; params_kwargs...)
            update_mll!(gp)
            update_dmll!(gp; params_kwargs...)
            grad[:] = -gp.dmll
            return -gp.mll
        catch err
            # reset parameters to remove any NaNs
            GaussianProcesses.set_params!(gp, prev; params_kwargs...)
            if !all(isfinite.(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, LinearAlgebra.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end
    end

    function dltarget!(grad::AbstractVector, hyp::AbstractVector)
        ltarget_and_dltarget!(grad::AbstractVector, hyp::AbstractVector)
    end
    xinit = GaussianProcesses.get_params(gp; params_kwargs...)
    func = GaussianProcesses.OnceDifferentiable(ltarget, dltarget!, ltarget_and_dltarget!, xinit)
    return func
end

function GaussianProcesses.init_precompute(gp::GPLA) nothing end
