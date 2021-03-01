"""
Monte Carlo Planning solver with DPW and Bayesian Optimization Action Selection

Fields:

    depth::Int64:
        Maximum rollout horizon and tree depth.
        default: 10

    exploration_constant::Float64:
        Specified how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0

    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf

    k_action::Float64
    alpha_action::Float64
    k_state::Float64
    alpha_state::Float64
        These constants control the double progressive widening. A new state
        or action will be added if the number of children is less than or equal to kN^alpha.
        defaults: k:10, alpha:0.5

    check_repeat_state::Bool
    check_repeat_action::Bool
        When constructing the tree, check whether a state or action has been seen before (there is a computational cost to maintaining the dictionaries necessary for this)
        default: true

    tree_in_info::Bool:
        If true, return the tree in the info dict when action_info is called. False by default because it can use a lot of memory if histories are being saved.
        default: false

    rng::AbstractRNG:
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, depth)` will be called to estimate the value.
        If this is an object `o`, `estimate_value(o, mdp, s, depth)` will be called.
        If this is a number, the value will be set to that number.
        default: RolloutEstimator(RandomSolver(rng))

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will always be set to that number.
        default: 0.0

    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will always be set to that number.
        default: 0

    default_action::Any
        Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
        If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
        If this is a Policy `p`, `action(p, belief)` will be called.
        If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.
        default: `ExceptionRethrow()`
"""
mutable struct BOMCPSolver <: Solver
    action_selector::ActionSelector
    belief_updater::Updater
    depth::Int64
    exploration_constant::Float64
    n_iterations::Int64
    max_time::Float64
    k_action::Float64
    alpha_action::Float64
    k_belief::Float64
    alpha_belief::Float64
    tree_in_info::Bool
    rng::AbstractRNG
    estimate_value::Any
    init_Q::Any
    init_N::Any
    default_action::Any
end

"""
    BOMCPSolver()

Use keyword arguments to specify values for the fields
"""
function BOMCPSolver(
                    action_selector::ActionSelector,
                    belief_updater::Updater
                    ;depth::Int=10,
                    exploration_constant::Float64=1.0,
                    n_iterations::Int=100,
                    max_time::Float64=Inf,
                    k_action::Float64=10.0,
                    alpha_action::Float64=0.5,
                    k_belief::Float64=10.0,
                    alpha_belief::Float64=0.5,
                    tree_in_info::Bool=false,
                    rng::AbstractRNG=Random.GLOBAL_RNG,
                    estimate_value::Any = RolloutEstimator(RandomSolver(rng)),
                    init_Q::Any = 0.0,
                    init_N::Any = 0,
                    default_action::Any = ExceptionRethrow()
                   )
    BOMCPSolver(action_selector, belief_updater, depth, exploration_constant, n_iterations, max_time, k_action, alpha_action, k_belief, alpha_belief, tree_in_info, rng, estimate_value, init_Q, init_N, default_action)
end

mutable struct BOMCPPlanner{P<:Union{MDP,POMDP}, B, A, SE, RNG} <: Policy
    solver::BOMCPSolver
    p::P
    tree::Union{Nothing, BOMCPTree{B,A}}
    solved_estimate::SE
    rng::RNG
    gp::Union{Dict, GPLA}
end

function BOMCPPlanner(solver::BOMCPSolver, p::P) where P<:Union{POMDP,MDP}
    se = convert_estimator(solver.estimate_value, solver, p)
    gp = solver.action_selector.gp
    return BOMCPPlanner{P,
                      BOMCP.belief_type(solver.belief_updater),
                      actiontype(P),
                      typeof(se),
                      typeof(solver.rng)}(solver,
                                          p,
                                          nothing,
                                          se,
                                          solver.rng,
                                          gp
                     )
end

Random.seed!(p::BOMCPPlanner, seed) = Random.seed!(p.rng, seed)
POMDPs.solve(solver::BOMCPSolver, p::Union{POMDP,MDP}) = BOMCPPlanner(solver, p)
