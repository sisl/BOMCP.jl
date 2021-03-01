"""
Belief Updater that returns a Gaussian distribution centered at
prior observation with fixed covariance matrix
"""
struct GaussianObsUpdater <: Updater
    Σ::Array{Float64}
end

function belief_type(::GaussianObsUpdater)
    return MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},Array{Float64,1}}
end

function POMDPs.update(up::GaussianObsUpdater, ::Any, ::Any, o::O) where O
    return MvNormal(o, up.Σ)
end
"""
Belief Updater that implements an Extended Kalman Filter (EKF)
"""
struct EKFUpdater <: Updater
    m::POMDP
    Q::Matrix{Float64}
    R::Matrix{Float64}
    function EKFUpdater(m::POMDP, Q::Array, R::Array)
        if ndims(Q) == 1
            Q = diagm(Q)
        end
        if ndims(R) == 1
            R = diagm(R)
        end
        new(m, Q, R)
    end
end

function x2s(m::POMDP, x::Vector)
    return s
end

function s2x(m::POMDP, s::Any)
    return x
end

function gen_A(m::POMDP, s, a)
    return A
end

function gen_C(m::POMDP, s)
    return C
end

function belief_type(::EKFUpdater)
    return MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}
end

function POMDPs.update(up::EKFUpdater, b::MvNormal, a::Any, o::O) where O
    μ = mean(b)
    n = length(μ)
    Σ = cov(b)
    s = x2s(up.m, μ)
    # Predict
    sp, z = POMDPs.gen(DDNOut(:sp, :o), up.m, s, a, Random.GLOBAL_RNG)
    xp = s2x(up.m, sp)

    A = gen_A(up.m, μ, a)
    C = gen_C(up.m, xp)

    Σ_hat = A*Σ*transpose(A) + up.Q

    # Update
    y = o - z

    S = C*Σ_hat*transpose(C) + up.R
    K = Σ_hat*transpose(C)/S

    μp = xp + K*y

    Σp = (Matrix{Float64}(I, n, n) - K*C)*Σ_hat

    Σp = round.(Σp, digits=5)
    bp = MvNormal(μp, Σp)
    return bp
end
