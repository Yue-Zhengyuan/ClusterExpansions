"""Grow oriented polyominoes with a reusable frontier and fixed-size translation keys."""
function _grow_clusters(previous, ::Val{N}, trees_only::Bool) where {N}
    clusters = Vector{LatticeSite}[]
    seen = Set{NTuple{N, LatticeSite}}()
    considered = Set{LatticeSite}()
    frontier = LatticeSite[]
    candidate = Vector{LatticeSite}(undef, N)
    for sites in previous
        length(sites) == N - 1 || throw(ArgumentError("Previous clusters must contain $(N - 1) sites"))
        empty!(considered)
        union!(considered, sites)
        empty!(frontier)
        for (x, y) in sites
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                neighbor in considered && continue
                push!(considered, neighbor)
                push!(frontier, neighbor)
            end
        end
        x0 = minimum(first, sites)
        y0 = minimum(last, sites)
        for neighbor in frontier
            if trees_only
                attachments = count(site -> abs(site[1] - neighbor[1]) + abs(site[2] - neighbor[2]) == 1, sites)
                attachments == 1 || continue
            end
            shift_x = min(x0, neighbor[1])
            shift_y = min(y0, neighbor[2])
            for i in eachindex(sites)
                x, y = sites[i]
                candidate[i] = (x - shift_x, y - shift_y)
            end
            candidate[N] = (neighbor[1] - shift_x, neighbor[2] - shift_y)
            sort!(candidate)
            key = ntuple(i -> candidate[i], Val(N))
            key in seen && continue
            push!(seen, key)
            push!(clusters, copy(candidate))
        end
    end
    return clusters
end

"""Grow connected clusters by one site while deduplicating translations only."""
function get_nontrivial_terms(N::Integer; prev_clusters = [[(0, 0)]])
    N >= 1 || throw(ArgumentError("Cluster size must be positive"))
    N == 1 && return [[(0, 0)]]
    return _grow_clusters(prev_clusters, Val(Int(N)), false)
end

"""Order clusters so straight paths establish shared bond gauges before bent paths reuse them."""
function _cluster_solve_order(cluster::Cluster)
    corners = count(cluster.levels_sites) do levels
        (levels[1] != 0 || levels[3] != 0) && (levels[2] != 0 || levels[4] != 0)
    end
    return (cluster.m, cluster.n, corners)
end

"""Precompute cluster geometry once, retaining every spatial orientation."""
function generate_clusters(p::Integer; solving_loops::Bool = true, levels_convention = :tree_depth)
    1 <= p <= 9 || throw(ArgumentError("Cluster order must lie between 1 and 9"))
    Symbol(levels_convention) == :tree_depth || throw(ArgumentError("Only tree_depth virtual levels are supported"))
    previous = [[(0, 0)]]
    orders = Vector{Cluster}[]
    for N in 2:p
        candidates = _grow_clusters(previous, Val(Int(N)), !solving_loops)
        clusters = [_cluster_geometry(sites) for sites in candidates]
        sort!(clusters; by = _cluster_solve_order, alg = Base.Sort.MergeSort)
        push!(orders, clusters)
        previous = candidates
    end
    return orders
end
