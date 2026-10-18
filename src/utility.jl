"""Check tensor entries without assuming a particular sector or storage layout."""
_tensor_isfinite(tensor::AbstractTensorMap) = all(all(isfinite, block) for (_, block) in blocks(tensor))

"""Copy a tensor into the requested scalar type while preserving its spaces."""
function _convert_tensor(::Type{T}, tensor::AbstractTensorMap) where {T}
    return copy!(similar(tensor, T), tensor)
end
