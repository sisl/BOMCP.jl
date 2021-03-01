using D3Trees
using Colors

"""
Return text to display below the node corresponding to state or action s
"""
node_tag(s) = string(s)

"""
Return text to display in the tooltip for the node corresponding to state or action s
"""
tooltip_tag(s) = node_tag(s)

function D3Trees.D3Tree(policy::BOMCPPlanner, root_state; kwargs...)
    # check to see if visualization was enabled
    if !policy.solver.enable_tree_vis
        error("""
              Tree visualization was not enabled for this policy.

              Construct the solver with $(typeof(policy.solver))(enable_tree_vis=true, ...) to enable.
              """)
    end
    return D3Tree(policy.tree, root_state; kwargs...)
end

function D3Trees.D3Tree(policy::BOMCPPlanner; kwargs...)
    @warn("""
         D3Tree(planner::BOMCPPlanner) is deprecated and may be removed in the future. Instead, please use

             a, info = action_info(planner, state)
             D3Tree(info[:tree])

         Make sure that the tree_in_info solver option is set to true. You can also get this info from a POMDPToolbox History

             info = first(ainfo_hist(hist))
             D3Tree(info[:tree])
         """)
    return D3Tree(policy.tree; kwargs...)
end

function D3Trees.D3Tree(tree::BOMCPTree, root_state; title="BOMCP tree", kwargs...)
    if tree._vis_stats == nothing
        error("""
              Visualization was not enabled for this tree.

              Construct the BOMCP solver with 'enable_tree_vis=true' to enable.
              """)
    end

    vs = tree._vis_stats

    nsas = length(vs)
    nsa = length(tree.n_a)
    nodes = Vector{Dict{String, Any}}(undef, 1 + nsas + nsa)

    # root node
    if haskey(tree.state_map, root_state)
        root_id = tree.state_map[root_state]
    else
        error("Could not find state $root_state in tree for visualization.")
    end
    nodes[1] = Dict("type"=>:state,
                    "child_d3ids"=>[1+nsas+c for c in tree.child_ids[root_id]],
                    "tag"=>node_tag(root_state),
                    "tt_tag"=>tooltip_tag(root_state),
                    "n"=>tree.n_b[root_id],
                    "n_b"=>tree.n_b[root_id],
                    "parent_n"=>tree.n_b[root_id]
                   )

    # state-action nodes
    for i in 1:nsa
        a = tree.a_labels[i]
        nodes[1+nsas+i] = Dict("type"=>:action,
                               "child_d3ids"=>Int[],
                               "tag"=>node_tag(a),
                               "tt_tag"=>tooltip_tag(a),
                               "n"=>tree.n_a[i],
                               "q"=>tree.q[i],
                              )
    end

    # state-action-state nodes
    for (i,((said,sid),n)) in enumerate(vs)
        s = tree.b_labels[sid]
        nodes[1+i] = Dict("type"=>:state,
                          "child_d3ids"=>[1+nsas+c for c in tree.child_ids[sid]],
                          "tag"=>node_tag(s),
                          "tt_tag"=>tooltip_tag(s),
                          "n"=>n,
                          "n_b"=>tree.n_b[sid],
                          "parent_n"=>tree.n_a[said]
                         )
        # add as a child to corresponding sa node
        push!(nodes[1+nsas+said]["child_d3ids"], 1+i)

        n = n_b(StateNode(tree, sid))
        # add parent_n to all children
        for csan in children(StateNode(tree, sid))
            csaid = csan.id
            nodes[1+nsas+csaid]["parent_n"] = n
        end
    end

    for csan in children(StateNode(tree, root_id))
        csaid = csan.id
        nodes[1+nsas+csaid]["parent_n"] = n_b(StateNode(tree, root_id))
    end

    return D3Tree(nodes; title=title, kwargs...)
end

function D3Trees.D3Tree(nodes::Vector{Dict{String, Any}}; title="Julia D3Tree", kwargs...)
    len = length(nodes)
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tooltip = Vector{String}(undef, len)
    style = fill("", len)
    link_style = fill("", len)
    max_q = maximum(get(n, "q", 0.0) for n in nodes)
    min_q = minimum(get(n, "q", 0.0) for n in nodes)
    for i in 1:len
        n = nodes[i]
        children[i] = n["child_d3ids"]
        if n["type"] == :state
            text[i] = @sprintf("""
                               %25s
                               N: %6d
                               """,
                               n["tag"], n["n_b"])
            tooltip[i] = """
                         $(n["tt_tag"])
                         N: $(n["n_b"])
                         """
            w = 20.0*sqrt(n["n"]/n["parent_n"])
            link_style[i] = "stroke-width:$(w)px"
        elseif n["type"] == :action
            text[i] = @sprintf("""
                               %25s
                               Q: %6.2f
                               N: %6d
                               """,
                               n["tag"], n["q"], n["n"])
            tooltip[i] = """
                         $(n["tt_tag"])
                         Q: $(n["q"])
                         N: $(n["n"])
                         """

            rel_q = (n["q"]-min_q)/(max_q-min_q)
            color = weighted_color_mean(rel_q, colorant"green", colorant"red")
            style[i] = "stroke:#$(hex(color))"
            w = 20.0*sqrt(n["n"]/n["parent_n"])
            link_style[i] = "stroke-width:$(w)px"
        else
            @warn("Unrecognized node type when constructing D3Tree.")
        end
    end
    return D3Tree(children;
                  text=text,
                  tooltip=tooltip,
                  style=style,
                  link_style=link_style,
                  title=title,
                  kwargs...
                 )
end

function D3Trees.D3Tree(tree::BOMCPTree; title="BOMCP Tree", kwargs...)
    lens = length(tree.n_b)
    lensa = length(tree.n_a)
    len = lens + lensa
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tt = fill("", len)
    style = fill("", len)
    link_style = fill("", len)
    max_q = maximum(tree.q)
    min_q = minimum(tree.q)

    for s in 1:lens
        children[s] = tree.b_children[s] .+ lens
        text[s] =  @sprintf("""
                            %25s
                            N: %6d
                            """,
                            node_tag(tree.b_labels[s]),
                            tree.n_b[s]
                           )
        tt[s] = """
                $(tooltip_tag(tree.b_labels[s]))
                N: $(tree.n_b[s])
                """
        for sa in tree.b_children[s]
            w = 20.0*sqrt(tree.n_a[sa]/tree.n_b[s])
            link_style[sa+lens] = "stroke-width:$(w)px"
        end
    end
    for sa in 1:lensa
        children[sa+lens] = collect(first(t) for t in tree.a_children[sa])
        text[sa+lens] = @sprintf("""
                                 %25s
                                 Q: %6.2f
                                 N: %6d
                                 """,
                                 node_tag(tree.a_labels[sa]),
                                 tree.q[sa],
                                 tree.n_a[sa]
                                )
        tt[sa+lens] = """
                      $(tooltip_tag(tree.a_labels[sa]))
                      Q: $(tree.q[sa])
                      N: $(tree.n_a[sa])
                      """

        rel_q = (tree.q[sa]-min_q)/(max_q-min_q)
        if isnan(rel_q)
            color = colorant"gray"
        else
            color = weighted_color_mean(rel_q, colorant"green", colorant"red")
        end
        style[sa+lens] = "stroke:#$(hex(color))"
    end
    return D3Tree(children;
                  text=text,
                  tooltip=tt,
                  style=style,
                  link_style=link_style,
                  title=title,
                  kwargs...
                 )
end
