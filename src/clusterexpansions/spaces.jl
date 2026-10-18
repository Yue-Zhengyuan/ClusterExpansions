"""A callable virtual-space rule with concrete spaces and a cache indexed by level."""
struct _LevelSpaces{S <: ElementarySpace}
    base::S
    loop::S
    cache::Dict{Int, S}
end

"""Initialize tree levels zero and one and reserve a separate space for loop bonds."""
function _LevelSpaces(base::S, loop::S) where {S <: ElementarySpace}
    return _LevelSpaces(base, loop, Dict{Int, S}(0 => oneunit(base), 1 => base))
end

"""Return cached tree spaces or the shared space assigned to negative loop levels."""
function (spaces::_LevelSpaces)(level::Integer)
    level < 0 && return spaces.loop
    return get!(spaces.cache, Int(level)) do
        fuse(spaces.base^Int(level))
    end
end

"""Select a non-dual loop space of the same elementary-space type as the model."""
function _select_loop_space(pspace::S, loop_space, default::S) where {S <: ElementarySpace}
    selected = isnothing(loop_space) ? default : loop_space
    selected isa S || throw(ArgumentError("loop_space must have the same space type as the physical indices"))
    !isdual(selected) && dim(selected) > 0 || throw(ArgumentError("loop_space must be non-dual and have positive dimension"))
    return selected
end

"""Use full operator-Schmidt spaces for tree bonds of a local Hamiltonian."""
function _operator_spaces(pspace::ElementarySpace; loop_space = nothing)
    base = fuse(pspace ⊗ pspace')
    return _LevelSpaces(base, _select_loop_space(pspace, loop_space, base))
end

"""Choose default virtual spaces, preserving ten loop states for ungraded models."""
function _default_spaces(pspace::S) where {S <: ElementarySpace}
    loops = sectortype(S) == Trivial ? S(10) : nothing
    return _operator_spaces(pspace; loop_space = loops)
end
