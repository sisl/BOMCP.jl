abstract type ActionSelector end

rand_action(gp, actions) = rand(actions)

function build_gp(action_dims, belief_dims, mean_type, mean_params, kernel_type, kernel_params, log_noise, k_neighborhood)
    mean = isnothing(mean_params) ? mean_type() : mean_type(mean_params...)
    total_dims = action_dims + belief_dims
    kernel = isnothing(kernel_params) ? kernel_type() : kernel_type(kernel_params...)
    x = zeros(Float64, total_dims, 0)
    y = zeros(Float64, 0)
    GPLA(x, y, k_neighborhood, action_dims, belief_dims, mean, kernel, log_noise)
end

"""
Bayesian Optimization Action Selector
"""
mutable struct BOActionSelector{M, K, IA, GP, D} <: ActionSelector
    action_dims::Int64
    belief_dims::Int64
    discrete_actions::Bool
    k_neighbors::Int64
    mean_type::M
    mean_params::Union{Array, Nothing}
    kernel_type::K
    kernel_params::Union{Array, Nothing}
    belief_λ::Float64 # λ < 0 Use separate GPs
    log_noise::Float64
    buffer_size::Int64
    acquisition_function::Union{AbstractAcquisition, Function}
    lower_bounds::Union{Function, AbstractArray, Nothing}
    upper_bounds::Union{Function, AbstractArray, Nothing}
    optim::Union{NLopt.Opt, Nothing}
    initial_action::IA
    gp::GP
    b_a_dict::Dict
    obs_buffer::Tuple{Matrix{Float64}, Vector{Float64}} # Number of observations to store
    function BOActionSelector{M, K, IA, GP, D}(
                            gp::GP,
                            action_dims::Int64,
                            belief_dims::Int64,
                            discrete_actions::Bool;
                            k_neighbors::Int64 = 10,
                            mean_type::M = MeanConst,
                            mean_params::Union{Array, Nothing} = [0.],
                            kernel_type::K = SEIso,
                            kernel_params::Union{Array, Nothing} = [0. 0.],
                            belief_λ::Float64 = 1.0,
                            log_noise::Float64 = -3.0,
                            buffer_size::Int64 = 0,
                            acquisition_function = ExpectedImprovement(),
                            lower_bounds = nothing,
                            upper_bounds = nothing,
                            optim = nothing,
                            initial_action::IA = rand_action
                            ) where {M, K, IA, GP, D}
        obs_buffer = (zeros(Float64, action_dims + belief_dims, 0), zeros(Float64, 0),)
        if discrete_actions
            return new(action_dims, belief_dims, discrete_actions, k_neighbors, mean_type, mean_params, kernel_type, kernel_params, belief_λ, log_noise, buffer_size, acquisition_function, nothing, nothing, nothing, initial_action, gp, Dict(), obs_buffer)
        else
            if isnothing(lower_bounds) || isnothing(upper_bounds)
                throw(ArgumentError("Lower and upper bounds must be provided for continuous action spaces"))
            end
            if isnothing(optim)
                optim = NLopt.Opt(:LD_LBFGS, action_dims)
                setproperty!(optim, :maxeval, 10)
                NLopt.lower_bounds!(optim, lower_bounds)
                NLopt.upper_bounds!(optim, upper_bounds)
            end
            return new(action_dims, belief_dims, discrete_actions, k_neighbors, mean_type, mean_params, kernel_type, kernel_params, belief_λ, log_noise, buffer_size, acquisition_function, lower_bounds, upper_bounds, optim, initial_action, gp, Dict(), obs_buffer)
        end
    end
end

function BOActionSelector(action_dims::Int64,
                        belief_dims::Int64,
                        discrete_actions::Bool;
                        k_neighbors::Int64 = 10,
                        mean_type::Any = MeanConst,
                        mean_params::Union{Array, Nothing} = [0.],
                        kernel_type::Any = SEIso,
                        kernel_params::Union{Array, Nothing} = [0. 0.],
                        belief_λ::Float64 = 1.0,
                        log_noise::Float64 = -3.0,
                        buffer_size::Int64 = 0,
                        acquisition_function = ExpectedImprovement(),
                        lower_bounds = nothing,
                        upper_bounds = nothing,
                        optim = nothing,
                        initial_action::Any = rand_action,
                        )
    if belief_λ < 0
        gp = Dict{Int64, GPE}()
    else
        gp = build_gp(action_dims, belief_dims, mean_type, mean_params, kernel_type, kernel_params, log_noise, k_neighbors)
    end
    BOActionSelector{typeof(mean_type), typeof(kernel_type), typeof(initial_action), typeof(gp), discrete_actions}(gp,
                                                                                    action_dims,
                                                                                    belief_dims,
                                                                                    discrete_actions,
                                                                                    k_neighbors=k_neighbors,
                                                                                    mean_type=mean_type,
                                                                                    mean_params=mean_params,
                                                                                    kernel_type=kernel_type,
                                                                                    kernel_params=kernel_params,
                                                                                    belief_λ=belief_λ,
                                                                                    log_noise=log_noise,
                                                                                    buffer_size=buffer_size,
                                                                                    acquisition_function=acquisition_function,
                                                                                    lower_bounds=lower_bounds,
                                                                                    upper_bounds=upper_bounds,
                                                                                    optim=optim,
                                                                                    initial_action=initial_action
                                                                                    )
end
