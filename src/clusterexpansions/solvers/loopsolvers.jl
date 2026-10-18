"""Walk a plaquette in lattice order, starting at its first coordinate."""
function _plaquette_path(cluster::Cluster)
    cluster.N == 4 && cluster.m == 1 || throw(ArgumentError("The loop solver requires a single four-site plaquette"))
    graph = get_graph(cluster)
    all(count(!iszero, graph[site, :]) == 2 for site in 1:4) ||
        throw(ArgumentError("Each plaquette site must have exactly two neighbors"))
    path = [argmin(cluster.cluster)]
    for _ in 1:3
        candidates = [site for site in graph[path[end], :] if site != 0 && site ∉ path]
        sort!(candidates; by = site -> cluster.cluster[site])
        push!(path, first(candidates))
    end
    return path
end

"""Insert missing virtual legs and embed an SVD core into its PEPO block spaces."""
function _embed_loop_core(core::AbstractTensorMap, physical, incident, levels, spaces)
    directions = sort!(collect(keys(incident)))
    tensor = permute(core, (physical, Tuple(incident[direction] for direction in directions)))
    missing = setdiff(1:4, directions)
    for direction in missing
        tensor = insertleftunit(tensor, Val(numind(tensor) + 1); dual = direction > 2)
    end
    append!(directions, missing)
    order = ntuple(direction -> 2 + findfirst(==(direction), directions), 4)
    tensor = permute(tensor, ((1, 2), order))
    target = ntuple(direction -> direction > 2 ? spaces(levels[direction])' : spaces(levels[direction]), 4)
    embeddings = ntuple(direction -> isometry(scalartype(tensor), target[direction], domain(tensor)[direction]), 4)
    return tensor * foldl(⊗, embeddings)'
end

"""Initialize a plaquette by sequential SVDs with a rank-one closing bond."""
function _loop_svd_initial(RHS::AbstractTensorMap, cluster::Cluster, spaces, path)
    paired = Tuple(Iterators.flatten((site, site + 4) for site in path))
    remainder = permute(RHS, (paired, ()))
    T = scalartype(RHS)
    physical_space = codomain(RHS)[1]
    prototype = _empty_cluster_tensor(T, physical_space, first(cluster.levels_sites), spaces)
    tensors = Vector{typeof(prototype)}(undef, 4)
    for position in 1:3
        site = path[position]
        nextsite = path[position + 1]
        direction = first(get_direction(cluster.cluster[site], cluster.cluster[nextsite]))
        target = direction > 2 ? spaces(-1)' : spaces(-1)
        truncation_space = isdual(target) ? flip(target) : target
        nleft = position == 1 ? 2 : 3
        matrix = permute(remainder, (Tuple(1:nleft), Tuple((nleft + 1):numind(remainder))))
        U, S, V, _ = svd_trunc(matrix; trunc = truncspace(truncation_space))
        root = sqrt(S)
        core = U * root
        remainder = root * V
        if isdual(target)
            core = flip(core, numind(core))
            remainder = flip(remainder, 1)
        end
        embedding = isometry(scalartype(core), target, domain(core)[1])
        core = core * embedding'
        remainder = embedding * remainder
        if position == 1
            physical = (1, 2)
            incident = Dict(direction => 3)
        else
            previous = path[position - 1]
            previous_direction = first(get_direction(cluster.cluster[site], cluster.cluster[previous]))
            physical = (2, 3)
            incident = Dict(previous_direction => 1, direction => 4)
        end
        tensors[site] = _embed_loop_core(core, physical, incident, cluster.levels_sites[site], spaces)
    end
    site, previous = path[4], path[3]
    direction = first(get_direction(cluster.cluster[site], cluster.cluster[previous]))
    tensors[site] = _embed_loop_core(remainder, (2, 3), Dict(direction => 1), cluster.levels_sites[site], spaces)
    return tensors
end

"""Contract four plaquette tensors while projecting external bonds onto level zero."""
function _contract_loop(tensors, cluster::Cluster, spaces)
    indices = [[-site, -site - 4, 0, 0, 0, 0] for site in 1:4]
    for (label, (site₁, site₂)) in enumerate(cluster.bonds_indices)
        direction₁, direction₂ = get_direction(cluster.cluster[site₁], cluster.cluster[site₂])
        indices[site₁][direction₁ + 2] = label
        indices[site₂][direction₂ + 2] = label
    end
    T = scalartype(first(tensors))
    unit = spaces(0)
    direct, dual = ones(T, unit), ones(T, unit')
    boundary = typeof(direct)[]
    boundary_indices = Vector{Int}[]
    label = length(cluster.bonds_indices)
    for site in 1:4, direction in 1:4
        indices[site][direction + 2] == 0 || continue
        label += 1
        indices[site][direction + 2] = label
        push!(boundary, direction > 2 ? dual : direct)
        push!(boundary_indices, [label])
    end
    result = ncon(vcat(tensors, boundary), vcat(indices, boundary_indices))
    return permute(result, ((1, 2, 3, 4), (5, 6, 7, 8)))
end

"""Equalize local norms by scalar gauge changes that preserve their product."""
function _balance_loop!(tensors::AbstractVector{<:AbstractTensorMap})
    norms = norm.(tensors)
    all(value -> isfinite(value) && value > 0, norms) || return tensors
    common = exp(sum(log, norms) / length(norms))
    for site in eachindex(tensors)
        tensors[site] = tensors[site] * (common / norms[site])
    end
    return tensors
end

"""Fit one plaquette tensor by solving for a correction to the current iterate."""
function _loop_local_update(
        RHS::AbstractTensorMap, cluster::Cluster, dictionary::AbstractDict, site::Int;
        rtol::Real, linsolve_tol, linsolve_maxiter::Integer, verbosity::Integer
    )
    T = scalartype(RHS)
    environment = get_A(T, cluster, dictionary, [site])
    others = [other for other in 1:4 if other != site]
    target = permute(RHS, ((others..., (others .+ 4)...), (site, site + 4)))
    original = dictionary[cluster.levels_sites[site]]
    current = permute(original, ((3, 4, 5, 6), (1, 2)))
    action = (tensor, adjoint) -> apply_A_onesite(environment, tensor, 4, adjoint)
    residual = target - action(current, Val(false))
    initial_error = norm(residual)
    iszero(initial_error) && return original
    correction, info = _solve_cluster_equation(action, residual; rtol, linsolve_tol, linsolve_maxiter)
    candidate = current + correction
    _tensor_isfinite(candidate) || return original
    candidate_error = norm(target - action(candidate, Val(false)))
    if !isfinite(candidate_error) || candidate_error > initial_error
        return original
    end
    if info.converged == 0 && verbosity >= 3
        @info "Plaquette local solve reached its iteration limit" site iterations = info.numiter
    end
    return permute(candidate, ((5, 6), (1, 2, 3, 4)))
end

"""
    solve_4_loop(RHS, cluster, spaces; maxsweeps=12, tol=1e-10, kwargs...)

Approximate a four-site plaquette residual using sequential SVDs followed by alternating least-squares sweeps.
The returned tensors follow the cluster's site order, and the second result is their relative contraction error.
A fixed random seed makes the small perturbation used to activate the closing bond reproducible.
Set `maxsweeps=0` to return the SVD initialization alone.
"""
function solve_4_loop(
        RHS::AbstractTensorMap, cluster::Cluster, spaces;
        verbosity::Integer = 1, maxsweeps::Integer = 12, tol::Real = 1.0e-10,
        linsolve_maxiter::Integer = 200, linsolve_tol::Union{Nothing, Real} = nothing,
        perturbation::Real = 1.0e-2, seed::Integer = 0
    )
    path = _plaquette_path(cluster)
    maxsweeps >= 0 || throw(ArgumentError("maxsweeps must be nonnegative"))
    linsolve_maxiter > 0 || throw(ArgumentError("linsolve_maxiter must be positive"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    isfinite(perturbation) && perturbation >= 0 || throw(ArgumentError("perturbation must be finite and nonnegative"))
    length(codomain(RHS)) == 4 && length(domain(RHS)) == 4 ||
        throw(ArgumentError("The plaquette residual must have four output and four input physical indices"))
    all(level -> all(index -> index in (0, -1), level), cluster.levels_sites) ||
        throw(ArgumentError("A plaquette solver cannot update attached trees or higher loop levels"))
    isunitspace(spaces(0)) && dim(spaces(0)) == 1 ||
        throw(ArgumentError("The level-zero space must be a one-dimensional unit space"))
    !isdual(spaces(-1)) && dim(spaces(-1)) > 0 ||
        throw(ArgumentError("The plaquette loop space must be non-dual and have positive dimension"))
    ismonomorphic(spaces(0), spaces(-1)) ||
        throw(ArgumentError("The plaquette loop space must contain the trivial sector used by the SVD initialization"))
    _tensor_isfinite(RHS) || throw(ArgumentError("The plaquette residual contains nonfinite entries"))
    relative_local_tolerance = max(tol, eps(real(scalartype(RHS))))
    isnothing(linsolve_tol) || (isfinite(linsolve_tol) && linsolve_tol >= 0) ||
        throw(ArgumentError("linsolve_tol must be finite and nonnegative"))

    T = scalartype(RHS)
    physical = codomain(RHS)[1]
    all(==(physical), codomain(RHS)) && all(==(physical), domain(RHS)) ||
        throw(ArgumentError("The plaquette residual must use the same physical space at every site"))
    target_norm = norm(RHS)
    isfinite(target_norm) || throw(ArgumentError("The plaquette residual has a nonfinite norm"))
    if iszero(target_norm)
        tensors = [_empty_cluster_tensor(T, physical, levels, spaces) for levels in cluster.levels_sites]
        return tensors, zero(real(T))
    end
    tensors = _balance_loop!(_loop_svd_initial(RHS, cluster, spaces, path))
    best = copy.(tensors)
    best_error = norm(_contract_loop(best, cluster, spaces) - RHS) / target_norm
    (maxsweeps == 0 || best_error <= tol) && return best, best_error

    if !iszero(perturbation)
        rng = Xoshiro(seed)
        for site in eachindex(tensors)
            noise = randn(rng, scalartype(tensors[site]), codomain(tensors[site]), domain(tensors[site]))
            noise_norm = norm(noise)
            iszero(noise_norm) && continue
            tensors[site] += noise * (perturbation * norm(tensors[site]) / noise_norm)
        end
    end
    dictionary = Dict(cluster.levels_sites[site] => tensors[site] for site in 1:4)
    dictionary[(0, 0, 0, 0)] = _empty_cluster_tensor(T, physical, (0, 0, 0, 0), spaces)
    previous_error = norm(_contract_loop(tensors, cluster, spaces) - RHS) / target_norm
    for sweep in 1:maxsweeps
        order = isodd(sweep) ? (1:4) : (4:-1:1)
        for site in order
            tensors[site] = _loop_local_update(
                RHS, cluster, dictionary, site;
                rtol = relative_local_tolerance, linsolve_tol, linsolve_maxiter, verbosity
            )
            dictionary[cluster.levels_sites[site]] = tensors[site]
        end
        _balance_loop!(tensors)
        for site in 1:4
            dictionary[cluster.levels_sites[site]] = tensors[site]
        end
        relative_error = norm(_contract_loop(tensors, cluster, spaces) - RHS) / target_norm
        if isfinite(relative_error) && relative_error < best_error
            best = copy.(tensors)
            best_error = relative_error
        end
        verbosity >= 3 && @info "Plaquette sweep" sweep relative_error
        best_error <= tol && break
        isfinite(relative_error) || break
        previous_error - relative_error <= tol * max(previous_error, tol) && break
        previous_error = relative_error
    end
    if best_error > tol && verbosity > 0
        @warn "Plaquette loop approximation did not reach the requested tolerance" relative_error = best_error tolerance = tol
    end
    return best, best_error
end
