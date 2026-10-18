"""Return whether a lattice bond borders an elementary square on both sides."""
function _shared_plaquette_bond(a::NTuple{2, Int}, b::NTuple{2, Int}, occupied::Set{NTuple{2, Int}})
    dx, dy = b[1] - a[1], b[2] - a[2]
    return (a[1] - dy, a[2] + dx) in occupied &&
        (b[1] - dy, b[2] + dx) in occupied &&
        (a[1] + dy, a[2] - dx) in occupied &&
        (b[1] + dy, b[2] - dx) in occupied
end

"""Assign negative virtual levels to cyclic bonds while preserving attached tree levels."""
function update_levels_loops(
        levels::Vector{Int}, cycles::Vector{Vector{Int}}, m::Int,
        bonds_indices::Vector{Tuple{Int, Int}}, coordinates::Vector{NTuple{2, Int}}
    )
    m == length(cycles) || throw(ArgumentError("The number of loops does not match the cycle basis"))
    m == 0 && return levels
    cyclic_sites = BitSet()
    edges = Set{Tuple{Int, Int}}()
    for cycle in cycles
        union!(cyclic_sites, cycle)
        for i in eachindex(cycle)
            push!(edges, minmax(cycle[i], cycle[mod1(i + 1, length(cycle))]))
        end
    end
    ncyclic = length(cyclic_sites)
    supported = (m == 1 && ncyclic in (4, 8)) ||
        (m == 2 && ncyclic in (6, 7, 8)) ||
        (m == 3 && ncyclic in (8, 9)) || (m == 4 && ncyclic == 9)
    supported || throw(ArgumentError("Loop levels for $m loops on $ncyclic cyclic sites are not implemented"))

    occupied = Set(coordinates)
    separate_shared_bonds = m >= 3 || (m == 2 && ncyclic == 8)
    for (i, bond) in enumerate(bonds_indices)
        bond in edges || continue
        shared = separate_shared_bonds && _shared_plaquette_bond(coordinates[bond[1]], coordinates[bond[2]], occupied)
        levels[i] = shared ? -2 : -1
    end
    return levels
end
