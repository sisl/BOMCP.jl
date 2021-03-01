mutable struct BOMCPTree{B,A}
    ### for each belief node
    n_b::Vector{Int} # Number of visits, N(b)
    b_children::Vector{Vector{Int}} #Children, Ch(b)
    b_labels::Vector{B} # Labels, b
    v::Vector{Float64} # Value, V(b) (Unused)

    m_b::Vector{Int} # Number of visits from gen, B(bab')

    ### for each state-action node
    n_a::Vector{Int} #Number of times this action node has been visited
    a_children::Vector{Vector{Int}} #spnode
    p_nodes::Vector{Int} # The parent of this node (state not state ID)
    p_labels::Vector{B} # The parent of this node (state not state ID)
    a_labels::Vector{A} # The action associated with this node
    q::Vector{Float64} #Current Q estimate
    transitions::Dict{Tuple{Int64, B}, Tuple{Int, Int}} # Belief:(Bnode, n)
    r::Vector{Float64} # Mean reward


    function BOMCPTree{B,A}(sz::Int=1000) where {B,A}
        sz = min(sz, 100_000)
        return new(sizehint!(Int[], sz),
                   sizehint!(Vector{Int}[], sz),
                   sizehint!(B[], sz),
                   sizehint!(Float64[], sz),

                   sizehint!(Int[], sz),

                   sizehint!(Int[], sz),
                   sizehint!(Vector{Int}[], sz),
                   sizehint!(Int[], sz),
                   sizehint!(B[], sz),
                   sizehint!(A[], sz),
                   sizehint!(Float64[], sz),
                   Dict{Tuple{Int64, B}, Tuple{Int64,Int64,Float64}}(),
                   sizehint!(Float64[], sz)
                  )
    end
end

function insert_belief_node!(tree::BOMCPTree{B,A}, b::B) where {B,A}
    push!(tree.n_b, 0)
    push!(tree.b_children, Int[])
    push!(tree.b_labels, b)
    push!(tree.v, 0.0)

    push!(tree.m_b, 1)

    bnode = length(tree.n_b)
    return bnode
end

function insert_action_node!(tree::BOMCPTree{B,A}, bnode::Int, a::A, n0::Int, q0::Float64) where {B,A}
    push!(tree.n_a, n0)
    push!(tree.a_children, Int[])
    push!(tree.p_labels, tree.b_labels[bnode])
    push!(tree.p_nodes, bnode)
    push!(tree.a_labels, a)
    push!(tree.q, q0)
    push!(tree.r, 0.0)
    banode = length(tree.n_a)
    push!(tree.b_children[bnode], banode)
    return banode
end

Base.isempty(tree::BOMCPTree) = isempty(tree.n) && isempty(tree.q)

struct BOMCPBeliefNode{B,A}
    tree::BOMCPTree{B,A}
    index::Int
end

children(n::BOMCPBeliefNode) = n.tree.b_children[n.index]
n_children(n::BOMCPBeliefNode) = length(children(n))
isroot(n::BOMCPBeliefNode) = n.index == 1
