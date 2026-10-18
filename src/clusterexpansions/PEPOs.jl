"""Allocate a local PEPO tensor with the requested physical and virtual spaces."""
function _empty_cluster_tensor(::Type{T}, pspace::ElementarySpace, levels::VirtualLevels, spaces) where {T}
    virtual = ntuple(dir -> dir > 2 ? spaces(levels[dir])' : spaces(levels[dir]), 4)
    return zeros(T, pspace ⊗ pspace', ⊗(virtual...))
end

"""Initialize the purely on-site exponential in the empty virtual sector."""
function init_PEPO(::Type{T}, β::Number, trivspace::ElementarySpace, onesite_op::AbstractTensorMap) where {T}
    local_operator = scalartype(onesite_op) == T ? onesite_op : _convert_tensor(T, onesite_op)
    exp_H = exp(-β * local_operator)
    _tensor_isfinite(exp_H) || error("The on-site exponential exceeds the working scalar range")
    physical = permute(exp_H, ((1, 2), ()))
    boundary = permute(isomorphism(T, trivspace' ⊗ trivspace', trivspace' ⊗ trivspace'), ((), (1, 2, 3, 4)))
    A = physical * boundary
    return Dict((0, 0, 0, 0) => A)
end

"""Return the ordered direct sum of tree levels followed by loop levels."""
function get_sum_space(highest::Integer, lowest::Integer, spaces)
    summands = [spaces(0)]
    sizehint!(summands, highest + 1 - min(lowest, 0))
    for level in 1:highest
        push!(summands, spaces(level))
    end
    for level in -1:-1:lowest
        push!(summands, spaces(level))
    end
    return SumSpace(summands)
end

"""Find occupied tree and loop level bounds in each virtual direction."""
function _level_bounds(PEPO::AbstractDict)
    highest = ntuple(dir -> maximum(key[dir] for key in keys(PEPO)), 4)
    lowest = ntuple(dir -> minimum(key[dir] for key in keys(PEPO)), 4)
    return highest, lowest
end

"""Assemble solved local tensors in a common direct-sum virtual space."""
function get_PEPO(::Type{T}, pspace::ElementarySpace, PEPO::AbstractDict, spaces; boundary::NTuple{4, Bool} = (false, false, false, false), level_bounds = _level_bounds(PEPO)) where {T}
    highest, lowest = level_bounds
    total_spaces = ntuple(dir -> boundary[dir] ? SumSpace(spaces(0)) : get_sum_space(highest[dir], lowest[dir], spaces), 4)
    Oout = SumSpace(pspace) ⊗ SumSpace(pspace)'
    Oin = ⊗(ntuple(dir -> dir > 2 ? total_spaces[dir]' : total_spaces[dir], 4)...)
    O = spzeros(T, Oout ← Oin)
    for (levels, tensor) in PEPO
        any(dir -> boundary[dir] && !iszero(levels[dir]), 1:4) && continue
        indices = ntuple(dir -> levels[dir] < 0 ? highest[dir] - levels[dir] + 1 : levels[dir] + 1, 4)
        O[1, 1, indices...] = tensor
    end
    return O
end

"""
    evolution_operator(alg::ClusterExpansion, t; T_conv = alg.T)

Return the local iPEPO tensor approximating ``\\exp(-t H)``.
The result retains its overall scale, including the on-site exponential.
Pass `im * Δt` for real-time evolution; wrap the tensor in `InfinitePEPO` for layer compression.
"""
function evolution_operator(alg::ClusterExpansion, β::Number; T_conv::Type{<:Number} = alg.T)
    _, directsum = clusterexpansion(alg, β)
    tensor = convert(TensorMap, directsum)
    return scalartype(tensor) == T_conv ? tensor : _convert_tensor(T_conv, tensor)
end
