const _PEPOLayer = Union{PEPSKit.PEPOTensor, InfinitePEPO}
const _LayerTruncation = Union{Nothing, PEPSKit.LocalTruncation, TruncationStrategy}

"""Construct compatible fusers for each of the four virtual bonds of a layer product."""
function _layer_fusers(A::AbstractTensorMap, B::PEPSKit.PEPOTensor)
    T = promote_type(scalartype(A), scalartype(B))
    storage = TensorKit.promote_storagetype(T, A, B)
    return ntuple(4) do dir
        if dir <= 2
            pair = domain(A)[dir] ⊗ domain(B)[dir]
            return isomorphism(storage, pair, fuse(pair))
        else
            pair = domain(A)[dir]' ⊗ domain(B)[dir]'
            return isomorphism(storage, fuse(pair), pair)
        end
    end
end

"""Check that the output physical index of the first layer matches the next input."""
function _check_layer_physicalspace(A::AbstractTensorMap, B::PEPSKit.PEPOTensor)
    space(A, 1) == space(B, 2)' ||
        throw(SpaceMismatch("The output physical space of the first layer does not match the input of the second layer."))
    return nothing
end

"""Reject a stacked PEPO where a single contracted layer is required."""
function _check_single_layer(A::InfinitePEPO)
    size(A, 3) == 1 || throw(ArgumentError("Each PEPO must contain exactly one layer."))
    size(A, 1) > 0 && size(A, 2) > 0 || throw(ArgumentError("The PEPO spatial unit cell must be nonempty."))
    return nothing
end

"""Validate single-layer unit cells and every physical contraction before combining them."""
function _check_layer_unitcells(A::Union{InfinitePEPS, InfinitePEPO}, O::InfinitePEPO)
    _check_single_layer(O)
    A isa InfinitePEPO && _check_single_layer(A)
    (size(A, 1), size(A, 2)) == (size(O, 1), size(O, 2)) ||
        throw(DimensionMismatch("The spatial unit cells must have equal sizes."))
    for column in 1:size(A, 2), row in 1:size(A, 1)
        _check_layer_physicalspace(A[row, column], O[row, column])
    end
    return nothing
end

"""
    apply_PEPO_exact(A, O)

Apply `O` to `A` without truncation, contracting the physical index and fusing each pair of virtual indices.
For PEPO inputs the result represents `O * A`, so the arguments follow their order of application.
Accepts local PEPS/PEPO tensors, an `InfinitePEPS` and a single-layer `InfinitePEPO`, or two single-layer `InfinitePEPO`s with equal spatial unit cells.
The four virtual spaces may differ, and the scalar normalization is preserved.
"""
function apply_PEPO_exact(A::PEPSKit.PEPOTensor, O::PEPSKit.PEPOTensor)
    _check_layer_physicalspace(A, O)
    Fn, Fe, Fs, Fw = _layer_fusers(A, O)
    @tensor opt = true result[pout pin; n e s w] :=
        A[p pin; na ea sa wa] * O[pout p; no eo so wo] *
        Fn[na no; n] * Fe[ea eo; e] * Fs[s; sa so] * Fw[w; wa wo]
    return result
end

function apply_PEPO_exact(A::PEPSKit.PEPSTensor, O::PEPSKit.PEPOTensor)
    _check_layer_physicalspace(A, O)
    Fn, Fe, Fs, Fw = _layer_fusers(A, O)
    @tensor opt = true result[pout; n e s w] :=
        A[p; na ea sa wa] * O[pout p; no eo so wo] *
        Fn[na no; n] * Fe[ea eo; e] * Fs[s; sa so] * Fw[w; wa wo]
    return result
end

function apply_PEPO_exact(A::InfinitePEPO, O::InfinitePEPO)
    _check_layer_unitcells(A, O)
    return InfinitePEPO(map(apply_PEPO_exact, PEPSKit.unitcell(A), PEPSKit.unitcell(O)))
end

function apply_PEPO_exact(A::InfinitePEPS, O::InfinitePEPO)
    _check_layer_unitcells(A, O)
    operators = reshape(PEPSKit.unitcell(O), size(A))
    return InfinitePEPS(map(apply_PEPO_exact, PEPSKit.unitcell(A), operators))
end

"""Keep local truncation errors without retaining the bond projector tensors."""
function _layer_compression_info(info::NamedTuple, ::Val{true})
    horizontal = map(projectors -> projectors[3].ϵ, info.Pxs_info)
    vertical = map(projectors -> projectors[3].ϵ, info.Pys_info)
    truncation_error = max(maximum(horizontal), maximum(vertical))
    return (; truncation_error, horizontal, vertical)
end

"""Discard compression diagnostics without allocating per-bond error arrays."""
_layer_compression_info(::NamedTuple, ::Val{false}) = nothing

"""Compose layers exactly and attach a zero truncation error."""
function _combine_layers(A::_PEPOLayer, B::_PEPOLayer, ::Nothing, ::Val{info}) where {info}
    result = apply_PEPO_exact(A, B)
    diagnostics = info ? (; truncation_error = zero(real(scalartype(result)))) : nothing
    return result, diagnostics
end

"""Compress a product directly, avoiding materializing its enlarged virtual bonds."""
function _combine_layers(A::InfinitePEPO, B::InfinitePEPO, alg::PEPSKit.LocalTruncation, keep_info::Val)
    _check_layer_unitcells(A, B)
    # PEPSKit contracts the first tuple entry's incoming physical index with the second entry's outgoing index.
    result, info = PEPSKit.compress((B, A), alg)
    return result, _layer_compression_info(info, keep_info)
end

"""Reuse the unit-cell compressor for a homogeneous pair of local PEPO tensors."""
function _combine_layers(A::PEPSKit.PEPOTensor, B::PEPSKit.PEPOTensor, alg::PEPSKit.LocalTruncation, keep_info::Val)
    result, info = _combine_layers(InfinitePEPO(A), InfinitePEPO(B), alg, keep_info)
    return result[1, 1, 1], info
end

"""Construct a local compressor once, retaining an existing algorithm or exact contraction."""
_layer_truncation(truncation::Union{Nothing, PEPSKit.LocalTruncation}) = truncation

"""Translate a bond truncation strategy into PEPSKit's local compression algorithm."""
_layer_truncation(truncation::TruncationStrategy) = PEPSKit.LocalTruncation(truncation)

"""
    combine_layers(A, B; truncation=nothing, return_info=false)

Combine two PEPO layers in application order, returning one layer representing `B * A`.
The inputs may be two local PEPO tensors or two single-layer `InfinitePEPO`s with equal spatial unit cells.
By default the contraction is exact and virtual bond dimensions multiply.
Set `truncation` to a `TruncationStrategy` or `PEPSKit.LocalTruncation` to compress the product directly with PEPSKit's local bond projectors.
No normalization is performed.
With `return_info=true`, return `(layer, info)`; `info.truncation_error` is the largest local bond truncation error, and compressed products also report `info.horizontal` and `info.vertical` for every unit-cell bond.
These local errors do not bound the error of the infinite operator.
"""
function combine_layers(
        A::PEPSKit.PEPOTensor, B::PEPSKit.PEPOTensor;
        truncation::_LayerTruncation = nothing, return_info::Bool = false,
    )
    result, info = _combine_layers(A, B, _layer_truncation(truncation), Val(return_info))
    return return_info ? (result, info) : result
end

function combine_layers(
        A::InfinitePEPO, B::InfinitePEPO;
        truncation::_LayerTruncation = nothing, return_info::Bool = false,
    )
    result, info = _combine_layers(A, B, _layer_truncation(truncation), Val(return_info))
    return return_info ? (result, info) : result
end

"""Count a layer multiplication and retain diagnostics only when requested."""
function _combine_power_step(A::_PEPOLayer, B::_PEPOLayer, truncation, history)
    result, info = _combine_layers(A, B, truncation, Val(!isnothing(history)))
    isnothing(history) || push!(history, info)
    return result
end

"""Copy a local layer without sharing its tensor entries."""
_copy_layer(layer::PEPSKit.PEPOTensor) = copy(layer)

"""Copy every tensor because PEPSKit's container copy shares tensor entries."""
_copy_layer(layer::InfinitePEPO) = InfinitePEPO(map(copy, PEPSKit.unitcell(layer)))

"""Apply the original layer repeatedly, retaining only the current product."""
function _layer_power_sequential(layer::_PEPOLayer, n::Integer, truncation, history)
    result = n == 1 ? _copy_layer(layer) : layer
    multiplications = 0
    for _ in 2:n
        result = _combine_power_step(result, layer, truncation, history)
        multiplications += 1
    end
    return result, multiplications
end

"""Build a power from its binary digits using the current product and original layer."""
function _layer_power_squaring(layer::_PEPOLayer, n::Integer, truncation, history)
    result = n == 1 ? _copy_layer(layer) : layer
    multiplications = 0
    for bit in (ndigits(n; base = 2) - 2):-1:0
        result = _combine_power_step(result, result, truncation, history)
        multiplications += 1
        if isodd(n >> bit)
            result = _combine_power_step(result, layer, truncation, history)
            multiplications += 1
        end
    end
    return result, multiplications
end

"""
    combine_layers(layer, n; truncation=nothing, method=:sequential, return_info=false)

Construct the positive integer power `layer^n` as a single PEPO layer, preserving its scalar normalization.
Use `method=:sequential` to apply the original layer `n - 1` times, or `method=:squaring` for exponentiation by squaring with at most logarithmically many multiplications.
With truncation the two methods generally produce different approximations.
Only the current product and original layer are retained, and `n == 1` returns an independent copy of the input.
With `return_info=true`, return `(layer, (; method, multiplications, steps))`, where `steps` contains the local truncation diagnostics in multiplication order.
"""
function combine_layers(
        layer::_PEPOLayer, n::Integer;
        truncation::_LayerTruncation = nothing, method::Symbol = :sequential,
        return_info::Bool = false,
    )
    n >= 1 || throw(ArgumentError("The number of layers must be positive."))
    method in (:sequential, :squaring) || throw(ArgumentError("method must be :sequential or :squaring."))
    layer isa InfinitePEPO && _check_single_layer(layer)
    history = return_info ? NamedTuple[] : nothing
    algorithm = _layer_truncation(truncation)
    result, multiplications = if method === :sequential
        _layer_power_sequential(layer, n, algorithm, history)
    else
        _layer_power_squaring(layer, n, algorithm, history)
    end
    return return_info ? (result, (; method, multiplications, steps = history)) : result
end
