struct Cluster
    N::Int
    cluster
    bonds_sites
    bonds_indices
    diag_bonds_sites
    diag_bonds_indices
    levels_sites
    m
    n
end

function Cluster(cluster; levels_convention = "tree_depth")
    N = length(cluster)
    bonds_sites, bonds_indices = get_bonds(cluster)
    diag_bonds_sites, diag_bonds_indices = get_diag_bonds(cluster)

    # coordination_number = [sum(i .∈ bonds_indices) for i = 1:N]

    g = SimpleGraph(Graphs.SimpleEdge.(bonds_indices))
    g_dir = get_directed_graph(bonds_indices)

    longest_path, n = get_longest_path(g_dir, N)
    # longest_cycle, m = get_longest_cycle(g_dir)
    cycles = cycle_basis(g_dir)
    m = length(cycles)

    if m >= 1
        coo = get_coordination_number(bonds_indices, N)
        if levels_convention == "initial"
            levels = get_levels(longest_path, n, bonds_indices, coo)
        elseif levels_convention == "tree_depth"
            levels = get_tree_depths(g, bonds_indices, vcat(cycles...))
        else
            error("Levels convention $(levels_convention) not implemented")
        end
        levels = update_levels_loops(levels, cycles, m, bonds_indices, cluster)
        middle_cycle = nothing
        if m >= 3
            for (ic, cycle) in enumerate(cycles)
                if sum([check_connectedness(cycle, c) for c in cycles]) == 2
                    middle_cycle = ic
                    break
                end
            end
        end
        for (i, (u, v)) in enumerate(bonds_indices)
            if (m >= 3) && (u ∈ cycles[middle_cycle] && v ∈ cycles[middle_cycle]) # ladder
                isedge = false
                for ic in setdiff(1:m, middle_cycle)
                    if u ∈ cycles[ic] && v ∈ cycles[ic]
                        isedge = true
                        break
                    end
                end
                levels[i] = -1 - isedge
            elseif u ∈ cycles[1] && v ∈ cycles[1] # all other cycles
                levels[i] = -1
            end
        end
        levels_sites = get_levels_sites(bonds_sites, bonds_indices, levels, N)
    else
        coo = get_coordination_number(bonds_indices, N)

        if levels_convention == "initial"
            levels = get_levels(longest_path, n, bonds_indices, coo)
        elseif levels_convention == "tree_depth"
            levels = get_tree_depths(g, bonds_indices)
        else
            error("Levels convention $(levels_convention) not implemented")
        end
        levels_sites = get_levels_sites(bonds_sites, bonds_indices, levels, N)
    end
    return Cluster(N, cluster, bonds_sites, bonds_indices, diag_bonds_sites, diag_bonds_indices, levels_sites, m, n)
end

function isdiagonal(site₁, site₂)
    return (abs(site₁[1] - site₂[1]) == 1) && (abs(site₁[2] - site₂[2]) == 1)
end

function distance(ind₁, ind₂)
    return abs(ind₁[1] - ind₂[1]) + abs(ind₁[2] - ind₂[2])
end

function get_direction(site₁, site₂)
    dir = (site₂[1] - site₁[1], site₂[2] - site₁[2])
    if dir == (-1, 0)
        return (1, 3)
    elseif dir == (0, 1)
        return (2, 4)
    elseif dir == (1, 0)
        return (3, 1)
    elseif dir == (0, -1)
        return (4, 2)
    end
end

function get_direction(dir::Int64)
    return 0
end

function get_direction(dir::Tuple{Int64, Int64})
    if dir == (-1, 0)
        return (1, 3)
    elseif dir == (0, 1)
        return (2, 4)
    elseif dir == (1, 0)
        return (3, 1)
    elseif dir == (0, -1)
        return (4, 2)
    end
end

function get_conjugated(dir)
    if dir == (-1, 0) || dir == (0, 1)
        return [Bool[0, 1, 1], Bool[0, 0, 1]]
    else
        return [Bool[0, 0, 1], Bool[0, 1, 1]]
    end
end

function get_bonds(cluster)
    bonds_sites = Vector{Tuple{Tuple{Int, Int}, Tuple{Int, Int}}}()
    bonds_indices = Vector{Tuple{Int, Int}}()
    for (i, ind₁) in enumerate(cluster)
        for (j, ind₂) in enumerate(cluster)
            if (j > i) && (distance(ind₁, ind₂) == 1)
                push!(bonds_sites, (ind₁, ind₂))
                push!(bonds_indices, (i, j))
            end
        end
    end
    return bonds_sites, bonds_indices
end

function get_diag_bonds(cluster)
    diag_bonds_sites = Vector{Tuple{Tuple{Int, Int}, Tuple{Int, Int}}}()
    diag_bonds_indices = Vector{Tuple{Int, Int}}()
    for (i, ind₁) in enumerate(cluster)
        for (j, ind₂) in enumerate(cluster)
            if (j > i) && isdiagonal(ind₁, ind₂)
                push!(diag_bonds_sites, (ind₁, ind₂))
                push!(diag_bonds_indices, (i, j))
            end
        end
    end
    return diag_bonds_sites, diag_bonds_indices
end

function get_coordination_number(bonds_indices, N)
    return [sum(i .∈ bonds_indices) for i in 1:N]
end

function get_directed_graph(bonds_indices)
    bonds_reversed = [(j, i) for (i, j) in bonds_indices]
    bonds_bi = unique(vcat(bonds_indices, bonds_reversed))
    return SimpleDiGraph(Graphs.SimpleEdge.(bonds_bi))
end

function get_longest_path(g_dir, N)
    longest_path_graph, n = find_longest_path(g_dir, N)
    longest_path = vcat([longest_path_graph[1].src], [l.dst for l in longest_path_graph])
    return longest_path, n
end

function find_longest_path(g, N)
    start_edge = 0
    max_dist = 0
    end_edge = 0
    for i in 1:N
        gdist = gdistances(g, i)
        dist = maximum(gdist)
        if dist > max_dist
            start_edge = i
            max_dist = dist
            end_edge = findall(x -> x == max_dist, gdist)
        end
    end
    return a_star(g, start_edge, end_edge[1]), max_dist + 1
end

function get_levels_sites(bonds_sites, bonds_indices, levels, N)
    levels_sites = fill(0, N, 4)

    for (i, (bond_s, bond_i)) in enumerate(zip(bonds_sites, bonds_indices))
        dir = get_direction(bond_s[1], bond_s[2])
        levels_sites[bond_i[1], dir[1]] = levels[i]
        levels_sites[bond_i[2], dir[2]] = levels[i]
    end

    levels_sites = [Tuple(levels_sites[i, :]) for i in 1:N]
    return levels_sites
end

# Pretty printing of a cluster
function Base.show(io::IO, z::Cluster)
    print(io, "Cluster =   ")
    # coords = [(x,y) for (x,y) in z.cluster]
    coords = [(y, -x) for (x, y) in z.cluster]

    # Determine the size of the graph
    min_x, max_x = minimum(x for (x, _) in coords), maximum(x for (x, _) in coords)
    min_y, max_y = minimum(y for (_, y) in coords), maximum(y for (_, y) in coords)

    offset = 6
    offset_cluster = 6
    max_grid = 20

    # grid = fill("  ", max_y - min_y + 1, offset + max_x - min_x + 1)
    grid = fill("  ", max_y - min_y + 1, max_grid)
    # Plot points
    for (x, y) in coords
        if y == max_y
            grid[max_y - y + 1, offset - offset_cluster + x - min_x + 1] = "● "
        else
            grid[max_y - y + 1, offset + x - min_x + 1] = "● "
        end
    end
    if z.m == 1
        grid[1, max_grid] = "has $(z.N) sites and $(z.m) loop"
    else
        grid[1, max_grid] = "has $(z.N) sites and $(z.m) loops"
    end
    # Print the graph in one call to avoid newline issues
    print(io, join([join(row) for row in eachrow(grid)], "\n"))
    return print(io, "\n")  # Keep the text on the same line but move to next before graph
end
