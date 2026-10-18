"""Find a tree's one or two centers and its diameter by pruning leaf layers."""
function _tree_centers(graph::SimpleGraph{Int})
    remaining = nv(graph)
    remaining == 1 && return [1], 0
    degrees = degree(graph)
    leaves = findall(==(1), degrees)
    layers = 0
    while remaining > 2
        remaining -= length(leaves)
        next_leaves = Int[]
        for leaf in leaves, neighbor in neighbors(graph, leaf)
            degrees[neighbor] -= 1
            degrees[neighbor] == 1 && push!(next_leaves, neighbor)
        end
        leaves = next_leaves
        layers += 1
    end
    return leaves, 2 * layers + (remaining == 2)
end

"""Return parent and subtree-depth arrays for a breadth-first spanning tree."""
function _rooted_tree_depths(graph::SimpleGraph{Int}, root::Int)
    parent = zeros(Int, nv(graph))
    parent[root] = root
    order = Int[root]
    next = 1
    while next <= length(order)
        current = order[next]
        next += 1
        for neighbor in neighbors(graph, current)
            if iszero(parent[neighbor])
                parent[neighbor] = current
                push!(order, neighbor)
            end
        end
    end
    depths = ones(Int, nv(graph))
    for node in Iterators.reverse(order)
        node == root && continue
        depths[parent[node]] = max(depths[parent[node]], depths[node] + 1)
    end
    return parent, depths
end

"""Minimize bond subtree depths over the selected tree centers or cycle roots."""
function get_tree_depths(graph::SimpleGraph{Int}, bonds_indices::Vector{Tuple{Int, Int}}, roots::Vector{Int})
    isempty(roots) && throw(ArgumentError("At least one tree root is required"))
    levels = fill(typemax(Int), length(bonds_indices))
    for root in roots
        parent, depths = _rooted_tree_depths(graph, root)
        for (i, (u, v)) in enumerate(bonds_indices)
            depth = parent[v] == u ? depths[v] : depths[u]
            levels[i] = min(levels[i], depth)
        end
    end
    return levels
end
