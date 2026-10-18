"""Add the missing local tensors needed to match a finite-cluster exponential."""
function solve_cluster!(PEPO::AbstractDict, cluster::Cluster, β::Number, alg::ClusterExpansion)
    alg.verbosity >= 2 && @info "Solving cluster" cluster
    updates = findall(levels -> !haskey(PEPO, levels), cluster.levels_sites)
    isempty(updates) && return PEPO
    !alg.solving_loops && cluster.m > 0 && return PEPO
    levels = cluster.levels_sites[updates]

    exp_H = exponentiate_hamiltonian(alg.twosite_op, alg.onesite_op, cluster, β)
    RHS = exp_H - contract_PEPO(alg.T, cluster, PEPO, alg.spaces)
    _tensor_isfinite(RHS) || error("Cluster residual contains nonfinite entries")
    target_norm, residual_norm = norm(exp_H), norm(RHS)
    isfinite(target_norm) && isfinite(residual_norm) || error("Cluster norms exceed the working scalar range")
    # Register zero blocks as solved so roundoff-sized corrections do not leave
    # missing keys in subsequent cluster environments.
    roundoff = 10 * eps(real(scalartype(RHS))) * target_norm
    if residual_norm <= roundoff
        pspace = domain(alg.onesite_op)[1]
        solutions = [_empty_cluster_tensor(alg.T, pspace, level, alg.spaces) for level in levels]
    elseif length(updates) == 4 && cluster.N == 4 && cluster.m == 1
        solutions, = solve_4_loop(RHS, cluster, alg.spaces; verbosity = alg.verbosity, maxsweeps = alg.loop_maxsweeps, tol = alg.loop_tol, linsolve_maxiter = alg.linsolve_maxiter, linsolve_tol = alg.linsolve_tol)
    elseif length(updates) in (1, 2)
        # The two-site cluster has no fixed tensors to contract into an environment.
        A = cluster.N == 2 ? nothing : get_A(alg.T, cluster, PEPO, updates)
        dir = length(updates) == 1 ? 0 : get_direction(cluster.cluster[updates[1]], cluster.cluster[updates[2]])
        solutions = solve_index(A, RHS, updates, levels, dir, alg.spaces; verbosity = alg.verbosity, linsolve_tol = alg.linsolve_tol, linsolve_maxiter = alg.linsolve_maxiter)
    else
        throw(ArgumentError("Solving $(length(updates)) new tensors on $(cluster.N) sites with $(cluster.m) loops is not supported; use a lower order or solving_loops = false"))
    end
    for (level, solution) in zip(levels, solutions)
        _tensor_isfinite(solution) || error("Cluster solver produced a nonfinite tensor at virtual levels $level")
        PEPO[level] = solution
    end
    return PEPO
end

"""Solve clusters in increasing size, with tree and shorter-path updates first."""
function _solve_clusters!(PEPO::AbstractDict, β::Number, alg::ClusterExpansion)
    for (order, clusters) in enumerate(alg.clusters)
        alg.verbosity >= 1 && @info "Cluster expansion" sites = order + 1 clusters = length(clusters)
        for cluster in clusters
            solve_cluster!(PEPO, cluster, β, alg)
        end
    end
    return PEPO
end

"""
    clusterexpansion(alg::ClusterExpansion, t)

Construct the local PEPO blocks and their direct-sum tensor for ``\\exp(-t H)``.
Return `(blocks, tensor)` without normalizing either representation.
For real-time evolution pass `t = im * Δt` and use a complex scalar type.
"""
function clusterexpansion(alg::ClusterExpansion, β::Number)
    isfinite(β) || throw(ArgumentError("The evolution parameter must be finite"))
    alg.T <: Real && !isreal(β) && throw(ArgumentError("A complex evolution parameter requires a complex working scalar type"))
    step = convert(alg.T, β)
    isfinite(step) || throw(ArgumentError("The evolution parameter overflows the working scalar type $(alg.T)"))
    PEPO = init_PEPO(alg.T, step, alg.spaces(0), alg.onesite_op)
    if !iszero(step) && !iszero(norm(alg.twosite_op))
        _solve_clusters!(PEPO, step, alg)
    end
    return PEPO, get_PEPO(alg.T, domain(alg.onesite_op)[1], PEPO, alg.spaces)
end

"""Construct a cluster expansion with an explicit working scalar type."""
function clusterexpansion(T::Type{<:Number}, p::Integer, β::Number, twosite_op::AbstractTensorMap, onesite_op::AbstractTensorMap; kwargs...)
    return clusterexpansion(ClusterExpansion(twosite_op, onesite_op; T, p, kwargs...), β)
end

"""Construct a cluster expansion using complex double precision by default."""
function clusterexpansion(p::Integer, β::Number, twosite_op::AbstractTensorMap, onesite_op::AbstractTensorMap; T::Type{<:Number} = ComplexF64, kwargs...)
    return clusterexpansion(T, p, β, twosite_op, onesite_op; kwargs...)
end
