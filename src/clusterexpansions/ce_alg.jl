"""
    ClusterExpansion(twosite_op, onesite_op; p = 3, kwargs...)

Configure a nearest-neighbor square-lattice cluster expansion of ``\\exp(-t H)``.
Cluster geometry is prepared once and reused for subsequent calls to `evolution_operator`.
The callable `spaces(level)` selects virtual sectors, with level zero equal to the tensor-unit space and negative levels reserved for loops.
Spaces needed by the cluster plan are checked when the algorithm is constructed.
Set `solving_loops = false` to retain only tree clusters.
Two-site tensors are split by SVD; the legacy `svd = false` option is unsupported.
`linsolve_tol` is an absolute tolerance for the normal-equation residual, and `linsolve_maxiter` limits linear-solver iterations.
`loop_tol` and `loop_maxsweeps` control the relative plaquette error and number of fitting sweeps.
"""
struct ClusterExpansion{H2, H1, F, S, L, R}
    twosite_op::H2
    onesite_op::H1
    p::Int
    verbosity::Int
    T::Type{S}
    spaces::F
    solving_loops::Bool
    linsolve_tol::L
    linsolve_maxiter::Int
    loop_tol::R
    loop_maxsweeps::Int
    clusters::Vector{Vector{Cluster}}
end

"""Validate the ranks, physical spaces, and finite entries of the local Hamiltonian."""
function _check_local_operators(twosite_op::AbstractTensorMap, onesite_op::AbstractTensorMap)
    numout(onesite_op) == numin(onesite_op) == 1 || throw(ArgumentError("onesite_op must be a one-site operator"))
    codomain(onesite_op) == domain(onesite_op) || throw(ArgumentError("onesite_op must be square"))
    pspace = domain(onesite_op)[1]
    codomain(twosite_op) == domain(twosite_op) == pspace ⊗ pspace || throw(ArgumentError("twosite_op must act on two copies of the one-site physical space"))
    _tensor_isfinite(twosite_op) && _tensor_isfinite(onesite_op) || throw(ArgumentError("Local Hamiltonian operators must have finite entries"))
    return pspace
end

"""Validate and warm up the virtual-space rule for every level needed by a plan."""
function _check_expansion_spaces(spaces, pspace::S, clusters) where {S <: ElementarySpace}
    highest, lowest = 0, 0
    for order in clusters, cluster in order, levels in cluster.levels_sites, level in levels
        highest = max(highest, level)
        lowest = min(lowest, level)
    end
    for level in Iterators.flatten((0:highest, -1:-1:lowest))
        virtual = spaces(level)
        virtual isa S || throw(ArgumentError("spaces($level) must have the same elementary-space type as the physical indices"))
        !isdual(virtual) && dim(virtual) > 0 || throw(ArgumentError("spaces($level) must be non-dual and have positive dimension"))
        if iszero(level)
            virtual == oneunit(pspace) || throw(ArgumentError("spaces(0) must be the tensor-unit space"))
        elseif level == -1
            ismonomorphic(oneunit(pspace), virtual) || throw(ArgumentError("The plaquette loop space must contain the trivial sector"))
        end
    end
    return nothing
end

function ClusterExpansion(
        twosite_op::AbstractTensorMap, onesite_op::AbstractTensorMap;
        p::Integer = 3, verbosity::Integer = 0, T::Type{<:Number} = ComplexF64,
        spaces = nothing, solving_loops::Bool = true, svd::Bool = true,
        linsolve_tol = nothing, linsolve_maxiter::Integer = 1000,
        loop_tol::Real = 1.0e-10, loop_maxsweeps::Integer = 12,
        levels_convention = :tree_depth,
    )
    twosite_op = convert(TensorMap, twosite_op)
    onesite_op = convert(TensorMap, onesite_op)
    pspace = _check_local_operators(twosite_op, onesite_op)
    isconcretetype(T) && (T <: AbstractFloat || T <: Complex{<:AbstractFloat}) || throw(ArgumentError("T must be a concrete floating-point scalar type"))
    svd || throw(ArgumentError("svd = false is no longer supported; use the general SVD splitting with svd = true"))
    verbosity >= 0 || throw(ArgumentError("verbosity must be nonnegative"))
    linsolve_maxiter > 0 || throw(ArgumentError("linsolve_maxiter must be positive"))
    isnothing(linsolve_tol) || (linsolve_tol isa Real && isfinite(linsolve_tol) && linsolve_tol > 0) || throw(ArgumentError("linsolve_tol must be finite and positive"))
    isfinite(loop_tol) && loop_tol > 0 || throw(ArgumentError("loop_tol must be finite and positive"))
    loop_maxsweeps >= 0 || throw(ArgumentError("loop_maxsweeps must be nonnegative"))
    clusters = generate_clusters(p; solving_loops, levels_convention)
    selected_spaces = isnothing(spaces) ? _default_spaces(pspace) : spaces
    _check_expansion_spaces(selected_spaces, pspace, clusters)
    h2, h1 = _convert_tensor(T, twosite_op), _convert_tensor(T, onesite_op)
    _tensor_isfinite(h2) && _tensor_isfinite(h1) || throw(ArgumentError("Hamiltonian entries overflow the working scalar type $T"))
    return ClusterExpansion(h2, h1, Int(p), Int(verbosity), T, selected_spaces, solving_loops, linsolve_tol, Int(linsolve_maxiter), loop_tol, Int(loop_maxsweeps), clusters)
end
