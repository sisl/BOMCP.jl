# BOMCP.jl
BOMCP is an online POMDP tree search solver that uses Bayesian optimization to select actions for expansion during tree progressive widening. For more information, see the paper at https://arxiv.org/abs/2010.03597.

BOMCP solves problems defined using the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface. BOMCP requires definition of a function to vectorize the actions or action + belief pairs for use in the Gaussian process covariance kernel. More details can be found below.  

## Installation

```julia
import Pkg
Pkg.add(https://github.com/sisl/BOMCP.jl)
```
## Usage
```julia 
using BOMCP

using POMDPs
using POMDPSimulators

using D3Trees
using Random

lander = LunarLander()
belief_updater = EKFUpdater(lander, lander.Q.^2, lander.R.^2)
rollout_policy = LanderPolicy(lander)

function BOMCP.vectorize!(v, dims, x::MvNormal)
    v = copy(mean(x))
    return v
end

action_selector = BOActionSelector(3, # action dims
                                6, #belief dims
                                false, #discrete actions
                                kernel_params=[log(5.0), 0.0],
                                k_neighbors = 5,
                                belief_λ = 0.5,
                                lower_bounds = [-10.0, 0.0, -0.5],
                                upper_bounds = [10.0, 15.0, 0.5],
                                buffer_size=100,
                                initial_action=rollout_policy
                                )


solver = BOMCPSolver(action_selector, belief_updater,
                    depth=250, n_iterations=100,
                    exploration_constant=50.0,
                    k_belief = 2.0,
                    alpha_belief = 0.1,
                    k_action = 3.,
                    alpha_action = 0.25,
                    estimate_value=BOMCP.RolloutEstimator(rollout_policy),
                    )

planner = POMDPs.solve(solver, lander)

b0 = POMDPs.initialstate_distribution(lander)
s0 = rand(b0)

hr = HistoryRecorder(max_steps=100)
hist = simulate(hr, lander, planner)
for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp
end
```
The algorithm behavior is determined by the keyword argument values passed to the solver constructor. The Bayesian optimized action selection behavior is defined by the `BOActionSelector` object which is defined in more detail in the following sub-section. The solver keyword arguments are shown below.

Fields:

- `eps::Float64`:
    Rollouts and tree expansion will stop when discount^depth is less than this.
    default: `0.01`
- `max_depth::Int`:
    Rollouts and tree expension will stop when this depth is reached.
    default: `10`
- `criterion::Any`:
    Criterion to decide which action to take at each node. e.g. `MaxUCB(c)`, `MaxQ`, or `MaxTries`.
    default: `MaxUCB(1.0)`
- `final_criterion::Any`:
    Criterion for choosing the action to take after the tree is constructed.
    default: `MaxQ()`
- `tree_queries::Int`:
    Number of iterations during each action() call.
    default: `100`
- `max_time::Float64`:
    Time limit for planning at each steps (seconds).
    default: `Inf`
- `rng::AbstractRNG`:
    Random number generator.
    default: `Base.GLOBAL_RNG`
- `node_sr_belief_updater::Updater`:
    Updater for state-reward distribution at the nodes.
    default: `POWNodeFilter()`
- `estimate_value::Any`: (rollout policy can be specified by setting this to RolloutEstimator(policy))
    Function, object, or number used to estimate the value at the leaf nodes.
    If this is a function `f`, `f(pomdp, s, h::BeliefNode, steps)` will be called to estimate the value.
    If this is an object `o`, `estimate_value(o, pomdp, s, h::BeliefNode, steps)` will be called.
    If this is a number, the value will be set to that number.
    default: `RolloutEstimator(RandomSolver(rng))`
- `enable_action_pw::Bool`:
    Controls whether progressive widening is done on actions; if `false`, the entire action space is used.
    default: `true`
- `check_repeat_obs::Bool`:
    Check if an observation was sampled multiple times. This has some dictionary maintenance overhead, but prevents multiple nodes with the same observation from being created. If the observation space is discrete, this should probably be used, but can be turned off for speed.
    default: `true`
- `check_repeat_act::Bool`:
    Check if an action was sampled multiple times. This has some dictionary maintenance overhead, but prevents multiple nodes with the same action from being created. If the action space is discrete, this should probably be used, but can be turned off for speed.
    default: `true`
- `k_action::Float64`, `alpha_action::Float64`, `k_observation::Float64`, `alpha_observation::Float64`:
    These constants control the double progressive widening. A new observation
    or action will be added if the number of children is less than or equal to kN^alpha.
    defaults: k: `10`, alpha: `0.5`
- `init_V::Any`:
    Function, object, or number used to set the initial V(h,a) value at a new node.
    If this is a function `f`, `f(pomdp, h, a)` will be called to set the value.
    If this is an object `o`, `init_V(o, pomdp, h, a)` will be called.
    If this is a number, V will be set to that number
    default: `0.0`
- `init_N::Any`:
    Function, object, or number used to set the initial N(s,a) value at a new node.
    If this is a function `f`, `f(pomdp, h, a)` will be called to set the value.
    If this is an object `o`, `init_N(o, pomdp, h, a)` will be called.
    If this is a number, N will be set to that number
    default: `0`
- `next_action::Any`
    Function or object used to choose the next action to be considered for progressive widening.
    The next action is determined based on the POMDP, the belief, `b`, and the current `BeliefNode`, `h`.
    If this is a function `f`, `f(pomdp, b, h)` will be called to set the value.
    If this is an object `o`, `next_action(o, pomdp, b, h)` will be called.
    default: `RandomActionGenerator(rng)`
- `default_action::Any`:
    Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
    If this is a Function `f`, `f(belief, ex)` will be called.
    If this is a Policy `p`, `action(p, belief)` will be called.
    If it is an object `a`, `default_action(a, belief, ex)` will be called, and
    if this method is not implemented, `a` will be returned directly.

## Bayesian Optimization Action Selector
The `BOActionSelector` object requires the action vector size and the belief vector size to be specified, along with whether or not the action space is dicrete. For discrete action spaces, the expected improvement is solved for at each action in the space. For continuous action spaces, an L-BFGS solver from [NLOpt.jl](https://github.com/JuliaOpt/NLopt.jl) is used. Additional keyword arguments control the behavior of the nearest-neighbor Gaussian process regression, defined below.
 
Fields:

- `k_neighbors::Int64`:
    Number of neighbors included in the local covariance kernel calculations.
    default: `10`

- `mean_type::Any`:
    Type of mean function to be used for Gaussian process prior. 
    Must conform to GaussianProcess.jl standards. Defaults to a constant mean prior. 
    default: `GaussianProcess.MeanConst`
    
- `mean_params::Any`:
    Parameters used to instantiate the `mean_type`.
    default: `[0.0]`

- `kernel_type::Any`:
    Type of kernel function to be used for Gaussian process prior. 
    Must conform to GaussianProcess.jl standards. 
    Defaults to a isometric squared-exponential function. 
    default: `GaussianProcess.SEIso`
    
- `kernel_params::Any`:
    Parameters used to instantiate the `kernel_type`.
    default: `[0.0, 0.0]`

- `belief_λ::Float64`:
    Weight used for the belief-action distance function. 
    Weight of 0.0 does not include belief in distance.
    In this case, only sibling action nodes are included in each GP regression.
    default: `1.0`

- `log_noise::Float64`:
    Log Gaussian process observation noise.
    default: `-3.0`
    
- `buffer_size::Int64`:
    Number of Gaussian process conditioning pairs to save at the end of tree search.
    Saved pairs will be used to condition the regression in subsequent tree searches.
    default: `0`
    
- `acquisition_function::Any`:
    Type of acquisition function to be used for Bayesian Optimization. 
    Must conform to BayesianOptimization.jl standards. 
    Defaults to Expected Improvment as defined in the paper. 
    default: `BayesianOptimization.ExpectedImprovement()`
    
- `lower_bounds::Union{Nothing, Vector{Float64}}`:
    Lower bounds on the action space required by the optimizer for continuous actions.
    If the default NLOpt L-BFGS optimizer is used, this is a required parameter.
    default: `nothing`
 
 - `upper_bounds::Union{Nothing, Vector{Float64}}`:
    Upper bounds on the action space required by the optimizer for continuous actions.
    If the default NLOpt L-BFGS optimizer is used, this is a required parameter.
    default: `nothing`
    
 - `opim::Any`:
    Optimizer to be called for continuous action spaces. 
    Must conform to the NLOpt.jl API. 
    If the action space is continous and `nothing` is specified, the L-BFGS solver is used.
    default: `nothing`
    
- `initial_action::Any`:
    Function or object to return for the intial action node selection. 
    Defaults to randomly sample an action from the action space.
    default: `rand_action`
