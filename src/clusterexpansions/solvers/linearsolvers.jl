"""Build boundary vectors with the orientation of each trivial virtual leg."""
function get_triv_tensors(::Type{T}, conjugated, trivspace::ElementarySpace) where {T}
    tensor = ones(T, trivspace)
    tensor_conj = ones(T, trivspace')
    return [conjugated_leg ? tensor_conj : tensor for conjugated_leg in conjugated]
end

"""Tabulate the neighboring site in each lattice direction, using zero at the boundary."""
function get_graph(cluster::Cluster)
    graph = zeros(Int, cluster.N, 4)
    for (bond_sites, bond_indices) in zip(cluster.bonds_sites, cluster.bonds_indices)
        dir = get_direction(bond_sites...)
        graph[bond_indices[1], dir[1]] = bond_indices[2]
        graph[bond_indices[2], dir[2]] = bond_indices[1]
    end
    return graph
end

"""Contract all known tensors into the environment of one site or two adjacent sites."""
function get_A(::Type{T}, cluster::Cluster, PEPO::AbstractDict, sites_to_update) where {T}
    updates = length(sites_to_update)
    updates in (1, 2) || throw(ArgumentError("the linear environment supports one or two updated sites"))
    updated = falses(cluster.N)
    for site in sites_to_update
        updated[site] = true
    end
    count(updated) == updates || throw(ArgumentError("updated sites must be distinct"))
    included_sites = findall(!, updated)
    fixed_tensors = length(included_sites)
    row_for_site = zeros(Int, cluster.N)
    row_for_site[included_sites] .= eachindex(included_sites)
    graph = get_graph(cluster)
    opposite = (3, 4, 1, 2)
    contraction_indices = zeros(Int, fixed_tensors, 6)
    for row in 1:fixed_tensors
        contraction_indices[row, 1] = -row
        contraction_indices[row, 2] = -(fixed_tensors + row)
    end

    next_index = 1
    for site in included_sites, dir in 1:4
        neighbor = graph[site, dir]
        row = row_for_site[site]
        if neighbor != 0 && !updated[neighbor] && contraction_indices[row, dir + 2] == 0
            contraction_indices[row, dir + 2] = next_index
            contraction_indices[row_for_site[neighbor], opposite[dir] + 2] = next_index
            next_index += 1
        end
    end
    number_of_bonds = next_index - 1
    conjugated = Bool[]
    for site in included_sites, dir in 1:4
        if graph[site, dir] == 0
            contraction_indices[row_for_site[site], dir + 2] = next_index
            push!(conjugated, dir > 2)
            next_index += 1
        end
    end

    open_conjugated = Bool[]
    open_indices = Int[]
    next_open = 2 * fixed_tensors + 1
    for site in sites_to_update, dir in 1:4
        neighbor = graph[site, dir]
        if neighbor != 0 && updated[neighbor]
            continue
        elseif neighbor == 0
            push!(open_conjugated, dir > 2)
            push!(open_indices, -next_open)
        else
            contraction_indices[row_for_site[neighbor], opposite[dir] + 2] = -next_open
        end
        next_open += 1
    end
    virtual_legs = next_open - 2 * fixed_tensors - 1
    virtual_legs == 2 * updates + 2 || throw(ArgumentError("the two updated sites must share a bond"))

    trivspace = domain(PEPO[(0, 0, 0, 0)])[1]
    nontriv_tensors = [PEPO[cluster.levels_sites[site]] for site in included_sites]
    triv_tensors = vcat(get_triv_tensors(T, conjugated, trivspace), get_triv_tensors(T, open_conjugated, trivspace))
    triv_contractions = vcat([[number_of_bonds + i] for i in eachindex(conjugated)], [[i] for i in open_indices])
    all_contractions = vcat([contraction_indices[row, :] for row in 1:fixed_tensors], triv_contractions)
    A = ncon(vcat(nontriv_tensors, triv_tensors), all_contractions)
    physical_legs = 2 * fixed_tensors
    return permute(A, (Tuple(1:physical_legs), Tuple((physical_legs + 1):(physical_legs + virtual_legs))))
end

"""Apply the one-site environment to a tensor with its virtual legs first."""
function apply_A_onesite(A::AbstractTensorMap, x::AbstractTensorMap, N::Integer, ::Val{false})
    fixed_legs = 2 * N - 2
    network = ((ntuple(i -> -i, fixed_legs)..., 1, 2, 3, 4), (1, 2, 3, 4, -fixed_legs - 1, -fixed_legs - 2))
    Ax = ncon((A, x), map(collect, network))
    return permute(Ax, (Tuple(1:fixed_legs), (fixed_legs + 1, fixed_legs + 2)))
end

"""Apply the adjoint one-site environment, including the fermionic twists."""
function apply_A_onesite(A::AbstractTensorMap, Ax::AbstractTensorMap, N::Integer, ::Val{true})
    fixed_legs = 2 * N - 2
    Ax′ = twist(Ax, N:fixed_legs)
    network = ((Tuple(1:fixed_legs)..., -1, -2, -3, -4), (Tuple(1:fixed_legs)..., -5, -6))
    x = ncon((A, Ax′), map(collect, network), (true, false))
    return permute(twist(x, (1, 2)), ((1, 2, 3, 4), (5, 6)))
end

"""Select the fermionic twists associated with the orientation of an updated bond."""
function get_twists(dir::Tuple{Int, Int})
    dir in ((3, 1), (4, 2)) && return (1, 2, 4)
    dir in ((1, 3), (2, 4)) && return (1, 4, 5)
    throw(ArgumentError("invalid bond directions: $dir"))
end

"""Apply the two-site environment to a tensor with its six external virtual legs first."""
function apply_A_twosite(A::AbstractTensorMap, x::AbstractTensorMap, N::Integer, dir, ::Val{false})
    fixed_legs = 2 * N - 4
    network = ((ntuple(i -> -i, fixed_legs)..., 1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6, -fixed_legs - 1, -fixed_legs - 2, -fixed_legs - 3, -fixed_legs - 4))
    Ax = ncon((A, x), map(collect, network))
    return permute(Ax, (Tuple(1:fixed_legs), Tuple((fixed_legs + 1):(fixed_legs + 4))))
end

"""Apply the adjoint two-site environment, including the orientation-dependent twists."""
function apply_A_twosite(A::AbstractTensorMap, Ax::AbstractTensorMap, N::Integer, dir::Tuple{Int, Int}, ::Val{true})
    fixed_legs = 2 * N - 4
    Ax′ = twist(Ax, (N - 1):fixed_legs)
    network = ((Tuple(1:fixed_legs)..., -1, -2, -3, -4, -5, -6), (Tuple(1:fixed_legs)..., -7, -8, -9, -10))
    x = ncon((A, Ax′), map(collect, network), (true, false))
    return permute(twist(x, get_twists(dir)), ((1, 2, 3, 4, 5, 6), (7, 8, 9, 10)))
end

"""Restore the physical-first PEPO convention after splitting a two-site tensor."""
function permute_dir(x::AbstractTensorMap, dir::Integer, second::Integer)
    1 <= dir <= 4 || throw(ArgumentError("a virtual direction must lie between one and four"))
    bond_index = 6 - 5 * second
    external_indices = ntuple(i -> i + 2 + second, 3)
    virtual_indices = ntuple(4) do i
        i == dir ? bond_index : external_indices[i - (i > dir)]
    end
    return permute(x, ((1 + second, 2 + second), virtual_indices))
end

"""Embed both factors in the requested bond space when truncation retains fewer states."""
function _pad_linear_bond(x1::AbstractTensorMap, x2::AbstractTensorMap, bondspace::ElementarySpace)
    retained_space = only(domain(x1))
    retained_space == bondspace && return x1, x2
    embedding = isometry(scalartype(x1), bondspace, retained_space)
    return x1 * embedding', embedding * x2
end

"""Solve a guarded least-squares equation with an absolute or relative normal-residual tolerance."""
function _solve_cluster_equation(
        apply_A, rhs::AbstractTensorMap;
        linsolve_tol = nothing, linsolve_maxiter::Integer = 1000,
        rtol::Real = eps(real(scalartype(rhs))), verbosity::Integer = 0
    )
    linsolve_maxiter > 0 || throw(ArgumentError("linsolve_maxiter must be positive"))
    isfinite(rtol) && rtol >= 0 || throw(ArgumentError("The relative linear tolerance must be finite and nonnegative"))
    isnothing(linsolve_tol) || (isfinite(linsolve_tol) && linsolve_tol >= 0) ||
        throw(ArgumentError("linsolve_tol must be finite and nonnegative"))
    _tensor_isfinite(rhs) || throw(ArgumentError("The linear right-hand side contains nonfinite entries"))
    rhs_norm = norm(rhs)
    normal_rhs = apply_A(rhs, Val(true))
    _tensor_isfinite(normal_rhs) || throw(ArgumentError("The linear environment produced nonfinite entries"))
    normal_norm = norm(normal_rhs)
    isfinite(rhs_norm) && isfinite(normal_norm) || throw(ArgumentError("The linear equation has a nonfinite norm"))
    tolerance = isnothing(linsolve_tol) ? rtol * normal_norm : linsolve_tol
    # KrylovKit normalizes both vectors before its first convergence check.
    # Handle stationary or zero equations before that normalization divides by zero.
    if iszero(rhs_norm) || iszero(normal_norm) || normal_norm <= tolerance
        x = zero(normal_rhs)
        info = (;
            converged = 1, relative_error = iszero(rhs_norm) ? zero(rhs_norm) : one(rhs_norm),
            normal_residual = normal_norm, numiter = 0, numops = 1,
        )
    else
        # Reuse the adjoint contraction already needed by the degeneracy check.
        initial = Ref{Union{Nothing, typeof(normal_rhs)}}(normal_rhs)
        apply_adjoint = function (tensor)
            cached = initial[]
            if tensor === rhs && !isnothing(cached)
                initial[] = nothing
                return cached
            end
            return apply_A(tensor, Val(true))
        end
        operator = (tensor -> apply_A(tensor, Val(false)), apply_adjoint)
        x, convergence = lssolve(operator, rhs, LSMR(; verbosity = 0, maxiter = linsolve_maxiter, tol = tolerance))
        info = (;
            converged = convergence.converged, relative_error = norm(convergence.residual) / rhs_norm,
            normal_residual = convergence.normres, numiter = convergence.numiter, numops = convergence.numops,
        )
    end
    if verbosity > 0 && (info.converged == 0 || info.relative_error > 1.0e-10)
        @warn "Cluster linear solve finished with relative residual $(info.relative_error)" converged = info.converged iterations = info.numiter
    end
    return x, info
end

"""Add the trivial external legs to an isolated two-site residual in its bond orientation."""
function _isolated_twosite_tensor(rhs::AbstractTensorMap, dir::Tuple{Int, Int}, trivspace::ElementarySpace)
    get_twists(dir)
    T = scalartype(rhs)
    matrix = permute(rhs, ((1, 3), (2, 4)))
    left_directions = Tuple(direction for direction in 1:4 if direction != dir[1])
    right_directions = Tuple(direction for direction in 1:4 if direction != dir[2])
    left_spaces = ⊗((direction > 2 ? trivspace : trivspace' for direction in left_directions)...)
    right_spaces = ⊗((direction > 2 ? trivspace' : trivspace for direction in right_directions)...)
    left = isomorphism(T, codomain(matrix) ⊗ left_spaces, codomain(matrix))
    right = isomorphism(T, domain(matrix), domain(matrix) ⊗ right_spaces)
    return left * matrix * right
end

"""Solve for one missing PEPO tensor or a bonded pair, then restore their leg conventions."""
function solve_index(
        A, exp_H::AbstractTensorMap, sites_to_update, levels_to_update, dir, spaces;
        verbosity::Integer = 0, linsolve_tol = nothing, linsolve_maxiter::Integer = 1000
    )
    N = numout(exp_H)
    updates = length(sites_to_update)
    updates in (1, 2) || throw(ArgumentError("a linear solve requires one or two updated tensors"))
    if N == 2 && updates == 2
        x = _isolated_twosite_tensor(exp_H, dir, spaces(0))
    else
        fixed_legs = 2 * (N - updates)
        virtual_legs = 2 * updates + 2
        A = permute(A, (Tuple(1:fixed_legs), Tuple((fixed_legs + 1):(fixed_legs + virtual_legs))))
        included_sites = setdiff(1:N, sites_to_update)
        rhs = permute(exp_H, ((included_sites..., (included_sites .+ N)...), (sites_to_update..., (sites_to_update .+ N)...)))
        apply_A = if updates == 1
            (tensor, adjoint) -> apply_A_onesite(A, tensor, N, adjoint)
        else
            (tensor, adjoint) -> apply_A_twosite(A, tensor, N, dir, adjoint)
        end
        x, _ = _solve_cluster_equation(apply_A, rhs; verbosity, linsolve_tol, linsolve_maxiter)
        iszero(norm(x)) && !iszero(norm(rhs)) &&
            error("The cluster linear solve returned zero for a nonzero residual; check the virtual spaces or tighten linsolve_tol")
        x = updates == 1 ? permute(x, ((5, 6), (1, 2, 3, 4))) : permute(x, ((7, 9, 1, 2, 3), (8, 10, 4, 5, 6)))
    end
    _tensor_isfinite(x) || error("The cluster linear solve returned a nonfinite tensor")
    updates == 1 && return [x]

    bondspace = spaces(levels_to_update[1][dir[1]])
    reverse_bond = dir in ((3, 1), (4, 2))
    # A later arrow flip must produce the requested dual space, including its charges.
    factor_space = reverse_bond ? flip(bondspace') : bondspace
    U, Σ, V, truncation_error = svd_trunc(x; trunc = truncspace(factor_space))
    sqrt_Σ = sqrt(Σ)
    x1, x2 = _pad_linear_bond(U * sqrt_Σ, sqrt_Σ * V, factor_space)
    solution_norm = norm(x)
    if verbosity > 0 && truncation_error > 100 * eps(real(scalartype(x))) * solution_norm
        @warn "Two-site SVD has relative truncation error $(truncation_error / solution_norm)"
    end
    x1 = permute_dir(x1, dir[1], 0)
    x2 = permute_dir(x2, dir[2], 1)
    if reverse_bond
        x1 = permute(flip(x1, 2 + dir[1]), ((1, 2), (3, 4, 5, 6)))
        x2 = permute(flip(x2, 2 + dir[2]), ((1, 2), (3, 4, 5, 6)))
    end
    return [x1, x2]
end
