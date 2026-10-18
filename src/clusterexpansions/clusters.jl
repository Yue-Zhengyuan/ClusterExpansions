const LatticeSite = NTuple{2, Int}
const VirtualLevels = NTuple{4, Int}

"""Geometry and virtual-level labels of a connected square-lattice cluster."""
struct Cluster
    N::Int
    cluster::Vector{LatticeSite}
    bonds_sites::Vector{Tuple{LatticeSite, LatticeSite}}
    bonds_indices::Vector{Tuple{Int, Int}}
    levels_sites::Vector{VirtualLevels}
    m::Int
    n::Int
end

"""Build the graph and tree-depth labels of a connected cluster."""
function Cluster(sites::AbstractVector; levels_convention = :tree_depth)
    Symbol(levels_convention) == :tree_depth || throw(ArgumentError("Only tree_depth virtual levels are supported"))
    return _cluster_geometry(LatticeSite[Tuple(site) for site in sites])
end

"""Build geometry from an owned coordinate vector without copying generated clusters."""
function _cluster_geometry(coordinates::Vector{LatticeSite})
    isempty(coordinates) && throw(ArgumentError("A cluster must contain at least one site"))
    allunique(coordinates) || throw(ArgumentError("Cluster sites must be distinct"))
    N = length(coordinates)
    bonds_sites, bonds_indices = get_bonds(coordinates)
    graph = SimpleGraph(N)
    for (i, j) in bonds_indices
        add_edge!(graph, i, j)
    end
    is_connected(graph) || throw(ArgumentError("Cluster sites must be connected by nearest-neighbor bonds"))
    N == 1 && return Cluster(N, coordinates, bonds_sites, bonds_indices, [(0, 0, 0, 0)], 0, 1)

    m = length(bonds_indices) - N + 1
    if iszero(m)
        roots, diameter = _tree_centers(graph)
        levels = get_tree_depths(graph, bonds_indices, roots)
        n = diameter + 1
    else
        cycles = cycle_basis(graph)
        roots = unique!(reduce(vcat, cycles))
        levels = get_tree_depths(graph, bonds_indices, roots)
        update_levels_loops(levels, cycles, m, bonds_indices, coordinates)
        n = maximum(maximum(gdistances(graph, site)) for site in vertices(graph)) + 1
    end
    return Cluster(N, coordinates, bonds_sites, bonds_indices, get_levels_sites(bonds_sites, bonds_indices, levels, N), m, n)
end

"""Map a nearest-neighbor displacement to the virtual legs at its endpoints."""
function get_direction(dir::LatticeSite)
    dir == (-1, 0) && return (1, 3)
    dir == (0, 1) && return (2, 4)
    dir == (1, 0) && return (3, 1)
    dir == (0, -1) && return (4, 2)
    throw(ArgumentError("Expected a nearest-neighbor displacement, got $dir"))
end

"""Return the pair of virtual legs joining two nearest-neighbor sites."""
get_direction(a::LatticeSite, b::LatticeSite) = get_direction((b[1] - a[1], b[2] - a[2]))

"""Enumerate nearest-neighbor bonds once using a coordinate lookup."""
function get_bonds(sites::AbstractVector{LatticeSite})
    locations = Dict{LatticeSite, Int}(site => i for (i, site) in enumerate(sites))
    bonds_sites = Tuple{LatticeSite, LatticeSite}[]
    bonds_indices = Tuple{Int, Int}[]
    for (i, (x, y)) in enumerate(sites)
        for neighbor in ((x + 1, y), (x, y + 1))
            j = get(locations, neighbor, 0)
            iszero(j) && continue
            push!(bonds_indices, minmax(i, j))
        end
    end
    sort!(bonds_indices)
    for (i, j) in bonds_indices
        push!(bonds_sites, (sites[i], sites[j]))
    end
    return bonds_sites, bonds_indices
end

"""Translate bond levels into the four ordered virtual legs at every site."""
function get_levels_sites(bonds_sites, bonds_indices, levels::AbstractVector{<:Integer}, N::Integer)
    labels = fill((0, 0, 0, 0), N)
    for (sites, indices, level) in zip(bonds_sites, bonds_indices, levels)
        directions = get_direction(sites...)
        labels[indices[1]] = Base.setindex(labels[indices[1]], level, directions[1])
        labels[indices[2]] = Base.setindex(labels[indices[2]], level, directions[2])
    end
    return labels
end

function Base.show(io::IO, cluster::Cluster)
    return print(io, "Cluster(", cluster.N, " sites, ", cluster.m, " loops, ", cluster.cluster, ")")
end
