"""
vectorize single input
"""

function vectorize!(v, dims, x::Union{Array, Real, <:Tuple{Vararg{Real}}})
    x = copy(x)
    if x isa Tuple
        n = length(x)
        for i = 1:n
            v[i] = convert(Float64, x[i])
        end
    elseif ndims(x) == 0
        v[1] = x
    else
        v = reshape(convert.(Float64, x), dims, 1)
    end
    return v
end

function vectorize!(v, dims, x::MvNormal)
    μ = mean(x)
    σ = diag(cov(x))
    dims_half = Int64(dims/2)
    v[1:dims_half] = μ
    v[dims_half+1:dims] = σ
    return v
end

function build_action_set(action_array::Array)
    n_actions = size(action_array)[end]
    action_set = Set()
    for i=1:n_actions
        array_slice = selectdim(action_array, ndims(action_array), i)
        union!(action_set, Set(array_slice))
    end
    return action_set
end

function acquire_max_action(opt, lowerbounds, upperbounds, restarts)
    maxf = -Inf
    maxx = lowerbounds
    seq = BayesianOptimization.ScaledLHSIterator(lowerbounds, upperbounds, restarts)
    for x0 in seq
        f, x, ret = NLopt.optimize(opt, x0)
        ret == NLopt.FORCED_STOP && throw(InterruptException())
        if f > maxf
            maxf = f
            maxx = x
        end
    end
    maxf, maxx
end

function action_function(acq, b, model)
    x -> begin
        x_test = [reshape(x, :, 1); b]
        μ, σ² = GaussianProcesses.predict_f(model, x_test)
        acq.(μ[1], max(σ²[1], 0.0))
    end
end

function next_action(o::BOActionSelector, p::Union{POMDP, MDP}, b, bnode)
    if haskey(o.b_a_dict, b)
        acts = o.b_a_dict[b]
    else
        if o.discrete_actions
            acts = build_action_set(POMDPs.actions(p, b))
        else
            acts = POMDPs.actions(p, b)
        end
        o.b_a_dict[b] = acts
    end
    b_node_id = bnode.index
    tree = bnode.tree
    if o.gp isa Dict
        if haskey(o.gp, b_node_id)
            gp = o.gp[b_node_id]
        else
            mean = isnothing(o.mean_params) ? o.mean_type() : o.mean_type(o.mean_params...)
            kernel = isnothing(o.kernel_params) ? o.kernel_type() : o.kernel_type(o.kernel_params...)
            x = zeros(Float64, o.action_dims, 0)
            y = zeros(Float64, 0)
            gp = GPE(x, y,
                    mean,
                    kernel,
                    o.log_noise)
            o.gp[b_node_id] = gp
        end
        belief_x = false
    else
        gp = o.gp
        belief_x = true
    end
    n_obs = size(gp.x, 2)
    if n_obs == 0
        if typeof(o.initial_action) <: Function
            action = o.initial_action(gp, acts)
        elseif typeof(o.initial_action) <: Policy
            action = POMDPs.action(o.initial_action, b)
        else
            action = o.initial_action in acts ? o.initial_action : throw(AssertionError("The provided initial action is not in the set returned by actions(pomdp, b)"))
        end
    else
        act_idxs = tree.b_children[b_node_id]
        obs_acts = tree.a_labels[act_idxs]
        obs_q = tree.q[act_idxs]
        if length(obs_q) >= 1
            o.acquisition_function.τ = maximum(obs_q)
        else
            o.acquisition_function.τ = -1e4 #This is the Lower Bound
        end
        if o.discrete_actions
            new_acts = collect(setdiff(acts, obs_acts))
            if isempty(new_acts)
                return rand(acts)
            end
            n_acts = length(new_acts)
            if belief_x
                vector_belief = zeros(Float64, o.belief_dims)
                vector_belief = vectorize!(vector_belief, o.belief_dims, b)
                vector_belief .*= o.belief_λ
                x = zeros(Float64, o.action_dims + o.belief_dims, n_acts)
                for i = 1:n_acts
                    x[1:o.action_dims,i] = vectorize!(x[1:o.action_dims,i], o.action_dims, new_acts[i])
                    x[o.action_dims+1:o.action_dims+o.belief_dims, i] = vector_belief
                end
            else
                x = zeros(Float64, o.action_dims, n_acts)
                for i = 1:n_acts
                    x[:,i] = vectorize!(x[:,i], o.action_dims, new_acts[i])
                end
            end
            mu, sig = GaussianProcesses.predict_f(gp, x)
            scores = o.acquisition_function.(mu, sig)
            act_idx = argmax(scores)
            action = new_acts[act_idx]
        else
            if o.gp isa Dict
                f = BayesianOptimization.wrap_gradient(BayesianOptimization.acquisitionfunction(o.acquisition_function, gp))
                NLopt.max_objective!(o.optim, f)
                _, action = acquire_max_action(o.optim, o.lower_bounds, o.upper_bounds, 100) # restarts)
            else
                vector_belief = zeros(Float64, o.belief_dims)
                vector_belief = vectorize!(vector_belief, o.belief_dims, b)
                vector_belief .*= o.belief_λ
                f = BayesianOptimization.wrap_gradient(action_function(o.acquisition_function, vector_belief, gp))
                NLopt.max_objective!(o.optim, f)
                _, action = acquire_max_action(o.optim, o.lower_bounds, o.upper_bounds, 100) # restarts)
            end
        end
    end
    return action
end

function update_gp!(gp::GPLA, tree, bnode, new_node, λ, buffer) #, dim::Val{M}, nobs::Val{N}) where {M, N}
    n_obs = length(tree.n_a)
    n_buffer = size(buffer[1], 2)
    tot_dim = beliefdim(gp) + actiondim(gp)
    if new_node
        gp.x = zeros(Float64, tot_dim, n_obs + n_buffer)
        gp.y = zeros(Float64, n_obs + n_buffer)
        for i = 1:n_obs
            gp.x[1:actiondim(gp), i] = vectorize!(gp.x[1:actiondim(gp), i], actiondim(gp), tree.a_labels[i])
            gp.x[actiondim(gp)+1:tot_dim, i] = vectorize!(gp.x[actiondim(gp)+1:tot_dim, i], beliefdim(gp), tree.p_labels[i])
            gp.x[actiondim(gp)+1:tot_dim, i] .*= λ
            gp.y[i] = tree.q[i] - tree.v[tree.p_nodes[i]]
        end
        gp.x[:,n_obs+1:n_buffer+n_obs] = buffer[1]
        gp.y[n_obs+1:n_buffer+n_obs] = buffer[2]
        gp.kdtree = KDTree(NearestNeighbors.copy_svec(Float64, gp.x, Val(tot_dim)))
    else
        for i = 1:n_obs
            gp.y[i] = tree.q[i] - tree.v[tree.p_nodes[i]]
        end
    end
    return gp
end

function update_gp!(gp::Dict, tree, bnode, new_node, λ)
    gpn = gp[bnode]
    act_idxs = tree.b_children[bnode]
    obs_acts = tree.a_labels[act_idxs]
    q_vals = tree.q[act_idxs]
    n_obs = length(obs_acts)
    x = zeros(Float64, gpn.dim, n_obs)
    for i = 1:n_obs
        x[:,i] = vectorize!(x[:,i], gpn.dim, obs_acts[i])
    end
    y = q_vals
    if new_node
        GaussianProcesses.fit!(gpn, x, y)
    else
        gpn.x = x
        gpn.y = y
        GaussianProcesses.update_mll!(gpn)
    end
    gp[bnode] = gpn
end

function update_gp_buffer!(action_selector::BOActionSelector)
    gp = action_selector.gp
    n = action_selector.buffer_size
    n_obs = size(gp.x, 2)
    if n_obs > n
        idxs = sample([1:n_obs;], n, replace=false)
        x_buffer = gp.x[:,idxs]
        y_buffer = gp.y[idxs]
    else
        x_buffer = gp.x
        y_buffer = gp.y
    end
    action_selector.obs_buffer = (x_buffer, y_buffer)
end

function reset_gp!(gp::GPLA)
    tot_dim = beliefdim(gp) + actiondim(gp)
    gp.x = zeros(Float64, tot_dim, 0)
    gp.y = zeros(Float64, 0)
    sx = NearestNeighbors.copy_svec(Float64, zeros(Float64, tot_dim, 0), Val(tot_dim))
    gp.kdtree = KDTree(sx, Euclidean())
end

function optimize_gp!(gp::GPLA, p)
    mean_results = nothing
    kernel_results = GaussianProcesses.optimize!(gp, method=Optim.ParticleSwarm(), domean = false, time_limit = 30.0,  allow_f_increases=true) #, g_tol=1e-7)
    return mean_results, kernel_results
end

function optimize_gp!(gp::Dict, p)
    x = zeros(Float64, size(gp[1].x, 1), 0)
    y = zeros(Float64, 0)
    for (key, value) in gp
        x = [x value.x]
        y = [y; value.y]
    end
    gpla = GPLA(x, y, 5, gp[1].dim, 0, gp[1].mean, gp[1].kernel, gp[1].logNoise.value)
    results = GaussianProcesses.optimize!(gpla, method=Optim.LBFGS(), time_limit = 60.0,  allow_f_increases=true)#,  g_tol=1e-4)

    mean_params = GaussianProcesses.get_params(gpla.mean)
    kernel_params = GaussianProcesses.get_params(gpla.kernel)
    noise_params = GaussianProcesses.get_params(gpla.logNoise)

    p.solver.action_selector.mean_params = mean_params
    p.solver.action_selector.kernel_params = kernel_params
    p.solver.action_selector.log_noise = noise_params[1]
    return results
end
