"""
Delete existing decision tree.
"""
function clear_tree!(p::BOMCPPlanner)
    p.tree = nothing
end

"""
Construct an BOMCP tree and choose the best action.
"""
POMDPs.action(p::BOMCPPlanner, b) = first(action_info(p, b))

"""
Construct an BOMCP tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(p::BOMCPPlanner, b; tree_in_info=false)
    local a::actiontype(p.p)
    info = Dict{Symbol, Any}()
    if p.gp isa GPLA
        reset_gp!(p.solver.action_selector.gp)
    end
    try
        B = typeof(b)
        A = actiontype(p.p)
        tree = BOMCPTree{B,A}(p.solver.n_iterations)
        p.tree = tree
        bnode = insert_belief_node!(tree, b)
        nquery = 0
        start_us = CPUtime_us()
        for i = 1:p.solver.n_iterations
            nquery += 1
            s = rand(p.rng, b)
            simulate(p, s, bnode, p.solver.depth)
            if p.gp isa GPLA
                update_gp_buffer!(p.solver.action_selector)
            end

            if CPUtime_us() - start_us >= p.solver.max_time * 1e6
                break
            end
        end
        backup_belief_node!(p.p, tree, 1)
        info[:search_time_us] = CPUtime_us() - start_us
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
        best_Q = -Inf
        banode = 0
        for child in tree.b_children[bnode]
            if tree.q[child] > best_Q
                best_Q = tree.q[child]
                banode = child
            end
        end
        a = tree.a_labels[banode] # choose action with highest approximate value
    catch ex
        a = convert(actiontype(p.p), default_action(p.solver.default_action, p.p, b, ex))
        info[:exception] = ex
    end
    return a, info
end


"""
Return the reward for one iteration of BOMCP.
    (p, s, bnode, p.solver.depth)
"""
function simulate(dpw::BOMCPPlanner, s, bnode::Int, d::Int)
    sol = dpw.solver
    tree = dpw.tree
    b = tree.b_labels[bnode]
    if isterminal(dpw.p, s)
        return 0.0
    elseif d == 0
        return estimate_value(dpw.solved_estimate, dpw.p, s, d)
    end

    # ========== Start action progressive widening ==========
    new_node = false
    if length(tree.b_children[bnode]) <= sol.k_action*tree.n_b[bnode]^sol.alpha_action
        a = next_action(sol.action_selector, dpw.p, b, BOMCPBeliefNode(tree, bnode)) # action generation step
        if !(a in tree.a_labels[tree.b_children[bnode]])
            n0 = init_N(sol.init_N, dpw.p, b, a)
            banode = insert_action_node!(tree, bnode, a, n0,
                                init_Q(sol.init_Q, dpw.p, b, a)
                               )
            tree.n_b[bnode] += n0
            new_node = true
        end
    end
    if !new_node
        best_UCB = -Inf
        banode = 0
        ltn = log(tree.n_b[bnode])
        for child in tree.b_children[bnode]
            n = tree.n_a[child]
            q = tree.q[child]
            c = sol.exploration_constant # for clarity
            UCB = q + c*sqrt(ltn/n)
            @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
            @assert !isequal(UCB, -Inf)
            if UCB > best_UCB
                best_UCB = UCB
                banode = child
                a = tree.a_labels[banode]
            end
        end
    end
    # ========== End action progressive widening ==========
    # ========== Start belief progressive widening ==========
    sp, o, r = POMDPs.gen(DDNOut(:sp, :o, :r), dpw.p, s, a, dpw.rng)
    bp = POMDPs.update(sol.belief_updater, b, a, o)
    if haskey(tree.transitions, (banode, bp))
        bpnode, n = tree.transitions[(banode, bp)]
        n += 1
        tree.transitions[(banode, bp)] = (bpnode, n)
        tree.m_b[bpnode] += 1
    elseif length(tree.a_children[banode]) <= sol.k_belief*tree.n_a[banode]^sol.alpha_belief
        bpnode = insert_belief_node!(tree, bp)
        n = 1
        tree.transitions[(banode, bp)] = (bpnode, n)
        push!(tree.a_children[banode], bpnode)
        new_node = true
    else
        children = tree.a_children[banode]
        weights = Weights(tree.m_b[children]/sum(tree.m_b[children]))
        bpnode = sample(dpw.rng, children, weights)
        bp = tree.b_labels[bpnode]
        bpnode, n = tree.transitions[(banode, bp)]
        sp = rand(bp)
        r = POMDPs.reward(dpw.p, s, a, sp)
    end
    # ========== End belief progressive widening ==========
    if new_node
        v = estimate_value(dpw.solved_estimate, dpw.p, sp, bp, d-1)
        tree.v[bpnode] = v
    else
        v = simulate(dpw, sp, bpnode, d-1)
    end

    tree.n_a[banode] += 1
    tree.n_b[bnode] += 1

    q = r + discount(dpw.p)*v
    tree.q[banode] += (q - tree.q[banode])/tree.n_a[banode]
    tree.r[banode] += (r - tree.r[banode])/tree.n_a[banode]

    tree.v[bnode] = maximum(tree.q[tree.b_children[bnode]])
    if sol.action_selector.gp isa GPLA
        update_gp!(sol.action_selector.gp, tree, bnode, new_node, sol.action_selector.belief_λ, sol.action_selector.obs_buffer) #, Val(sol.action_selector.gp.dim), Val(size(sol.action_selector.gp.x, 2)))
    else
        update_gp!(sol.action_selector.gp, tree, bnode, new_node, sol.action_selector.belief_λ)
    end
    return q
end

function backup_belief_node!(p::Union{POMDP, MDP}, tree::BOMCPTree, index::Int)
    children = tree.b_children[index]
    n_children = length(children)
    if n_children == 0
        return tree.v[index]
    else
        max_q = -Inf
        for i = 1:n_children
            child_id = children[i]
            q = backup_action_node!(p, tree, child_id)
            if q > max_q
                max_q = q
            end
        end
        tree.v[index] = max_q
        return max_q
    end
end

function backup_action_node!(p::Union{POMDP, MDP}, tree::BOMCPTree, index::Int)
    children = tree.a_children[index]
    n_children = length(children)
    q = 0.0
    m_total = 0.0
    for i = 1:n_children
        child_id = children[i]
        m = tree.m_b[child_id]
        q += backup_belief_node!(p, tree, child_id)*m
        m_total += m
    end
    q /= m_total
    q *= POMDPs.discount(p)
    q += tree.r[index]
    tree.q[index] = q
    return q
end
