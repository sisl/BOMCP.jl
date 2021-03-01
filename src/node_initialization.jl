"""
RolloutEstimator

If this is passed to the estimate_value field of the solver, rollouts will be used to estimate the value at the leaf nodes

Fields:
    solver::Union{Solver,Policy,Function}
        If this is a Solver, solve(solver, mdp) will be called to find the rollout policy
        If this is a Policy, the policy will be used for rollouts
        If this is a Function, a POMDPToolbox.FunctionPolicy with this function will be used for rollouts
"""
mutable struct RolloutEstimator
    solver::Union{Solver,Policy,Function} # rollout policy or solver
end

convert_to_policy(p::Policy, mdp::Union{POMDP,MDP}) = p
convert_to_policy(s::Solver, mdp::Union{POMDP,MDP}) = solve(s, mdp)
convert_to_policy(f::Function, mdp::Union{POMDP,MDP}) = FunctionPolicy(f)

struct PORollout
    solver::Union{POMDPs.Solver,POMDPs.Policy,Function}
    updater::POMDPs.Updater
end

struct SolvedPORollout{P<:POMDPs.Policy,U<:POMDPs.Updater,RNG<:AbstractRNG}
    policy::P
    updater::U
    rng::RNG
end

struct FORollout # fully observable rollout
    solver::Union{POMDPs.Solver,POMDPs.Policy}
end

struct SolvedFORollout{P<:POMDPs.Policy,RNG<:AbstractRNG}
    policy::P
    rng::RNG
end

struct FOValue
    solver::Union{POMDPs.Solver, POMDPs.Policy}
end

struct SolvedFOValue{P<:POMDPs.Policy}
    policy::P
end

"""
    estimate_value(estimator, problem::POMDPs.POMDP, start_state, h::BOMCPBeliefNode, steps::Int)

Return an initial unbiased estimate of the value at belief node h.

By default this runs a rollout simulation
"""
function estimate_value end
estimate_value(f::Function, pomdp::POMDPs.POMDP, start_state, b, steps::Int) = f(pomdp, start_state, h, steps)
estimate_value(n::Number, pomdp::POMDPs.POMDP, start_state, b, steps::Int) = convert(Float64, n)

function estimate_value(estimator::Union{SolvedPORollout,SolvedFORollout}, pomdp::POMDPs.POMDP, start_state, b, steps::Int)
    rollout(estimator, pomdp, start_state, b, steps)
end

@POMDP_require estimate_value(estimator::Union{SolvedPORollout,SolvedFORollout}, pomdp::POMDPs.POMDP, start_state, b, steps::Int) begin
    @subreq rollout(estimator, pomdp, start_state, b, steps)
end

function estimate_value(estimator::SolvedFOValue, pomdp::POMDPs.POMDP, start_state, b, steps::Int)
    POMDPs.value(estimator.policy, start_state)
end


function convert_estimator(ev::RolloutEstimator, solver, pomdp)
    policy = convert_to_policy(ev.solver, pomdp)
    SolvedPORollout(policy, updater(policy), solver.rng)
end

function convert_estimator(ev::PORollout, solver, pomdp)
    policy = convert_to_policy(ev.solver, pomdp)
    SolvedPORollout(policy, ev.updater, solver.rng)
end

function convert_estimator(est::FORollout, solver, pomdp)
    policy = convert_to_policy(est.solver, pomdp)
    SolvedFORollout(policy, solver.rng)
end

function convert_estimator(est::FOValue, solver::Solver, pomdp::POMDPs.POMDP)
    policy = convert_to_policy(est.solver, pomdp)
    SolvedFOValue(policy)
end


"""
Perform a rollout simulation to estimate the value.
"""
function rollout(est::SolvedPORollout, pomdp::POMDPs.POMDP, start_state, b, steps::Int)
    sim = RolloutSimulator(est.rng,
                           steps)
    return POMDPs.simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end

@POMDP_require rollout(est::SolvedPORollout, pomdp::POMDPs.POMDP, start_state, b, steps::Int) begin
    sim = RolloutSimulator(est.rng,
                           steps)
    @subreq POMDPs.simulate(sim, pomdp, est.policy, est.updater, b, start_state)
end


function rollout(est::SolvedFORollout, pomdp::POMDPs.POMDP, start_state, b, steps::Int)
    sim = RolloutSimulator(est.rng,
                                        steps)
    return POMDPs.simulate(sim, pomdp, est.policy, start_state)
end

@POMDP_require rollout(est::SolvedFORollout, pomdp::POMDPs.POMDP, start_state, b, steps::Int) begin
    sim = RolloutSimulator(est.rng,
                                        steps)
    @subreq POMDPs.simulate(sim, pomdp, est.policy, start_state)
end

"""
    init_Q(initializer, mdp, s, a)

Return a value to initialize Q(s,a) to based on domain knowledge.
"""
function init_Q end
init_Q(f::Function, mdp::Union{MDP,POMDP}, s, a) = f(mdp, s, a)
init_Q(n::Number, mdp::Union{MDP,POMDP}, s, a) = convert(Float64, n)

"""
    init_N(initializer, mdp, s, a)

Return a value to initialize N(s,a) to based on domain knowledge.
"""
function init_N end
init_N(f::Function, mdp::Union{MDP,POMDP}, s, a) = f(mdp, s, a)
init_N(n::Number, mdp::Union{MDP,POMDP}, s, a) = convert(Int, n)
