using Test
using Random
using LinearAlgebra
using TensorKit
using ClusterExpansions

const SolverCE = ClusterExpansions

"""Contract site-dependent PEPO factors from coordinates without using package environment builders."""
function solver_reference_contract(tensors::AbstractVector{<:AbstractTensorMap}, sites)
    N = length(sites)
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    labels = [[-site, -N - site, 0, 0, 0, 0] for site in 1:N]
    next_label = 0
    for first_site in 1:N, second_site in (first_site + 1):N
        displacement = (sites[second_site][1] - sites[first_site][1], sites[second_site][2] - sites[first_site][2])
        direction = findfirst(==(displacement), directions)
        isnothing(direction) && continue
        opposite = mod1(direction + 2, 4)
        next_label += 1
        labels[first_site][direction + 2] = next_label
        labels[second_site][opposite + 2] = next_label
    end
    network_tensors = AbstractTensorMap[tensors...]
    for site in 1:N, direction in 1:4
        iszero(labels[site][direction + 2]) || continue
        boundary_space = domain(tensors[site])[direction]
        @assert isunitspace(boundary_space) && dim(boundary_space) == 1
        next_label += 1
        labels[site][direction + 2] = next_label
        push!(network_tensors, ones(scalartype(tensors[site]), boundary_space))
        push!(labels, [next_label])
    end
    contracted = ncon(network_tensors, labels)
    return permute(contracted, (Tuple(1:N), Tuple((N + 1):(2 * N))))
end

"""Sample a local PEPO factor with explicit physical and virtual spaces."""
function solver_random_factor(rng::AbstractRNG, physical::ElementarySpace, levels, spaces)
    virtual = ntuple(direction -> direction > 2 ? spaces(levels[direction])' : spaces(levels[direction]), 4)
    return randn(rng, ComplexF64, physical ⊗ physical', ⊗(virtual...))
end

"""Check the defining adjoint identity for a cluster environment with prescribed updated sites."""
function solver_check_adjoint(rng::AbstractRNG, physical::ElementarySpace, virtual::ElementarySpace, sites, updates)
    cluster = SolverCE.Cluster(sites)
    unit = oneunit(physical)
    spaces = level -> iszero(level) ? unit : virtual
    tensors = [solver_random_factor(rng, physical, levels, spaces) for levels in cluster.levels_sites]
    dictionary = Dict(levels => tensor for (levels, tensor) in zip(cluster.levels_sites, tensors))
    dictionary[(0, 0, 0, 0)] = solver_random_factor(rng, physical, (0, 0, 0, 0), spaces)
    environment = SolverCE.get_A(ComplexF64, cluster, dictionary, updates)
    N = length(sites)
    physical_product = ⊗(ntuple(_ -> physical, N)...)
    full_rhs = randn(rng, ComplexF64, physical_product, physical_product)
    others = setdiff(1:N, updates)
    y = permute(full_rhs, ((others..., (others .+ N)...), (updates..., (updates .+ N)...)))
    action = if length(updates) == 1
        (tensor, flag) -> SolverCE.apply_A_onesite(environment, tensor, N, flag)
    else
        direction = SolverCE.get_direction(sites[updates[1]], sites[updates[2]])
        (tensor, flag) -> SolverCE.apply_A_twosite(environment, tensor, N, direction, flag)
    end
    prototype = action(y, Val(true))
    x = randn(rng, ComplexF64, codomain(prototype), domain(prototype))
    forward_inner = dot(action(x, Val(false)), y)
    adjoint_inner = dot(x, action(y, Val(true)))
    return @test isapprox(forward_inner, adjoint_inner; atol = 1.0e-11, rtol = 1.0e-11)
end

@testset "Cluster linear and plaquette solvers" begin
    rng = Xoshiro(71523)

    @testset "Degenerate least-squares equations" begin
        physical = ℂ^2
        matrix = TensorMap(ComplexF64[2 0; 0 0], physical, physical)
        action = (tensor, flag) -> flag === Val(true) ? matrix' * tensor : matrix * tensor
        supported = TensorMap(ComplexF64[6, 0], physical, one(physical))
        solution, info = SolverCE._solve_cluster_equation(action, supported; rtol = 1.0e-12, linsolve_maxiter = 20)
        @test solution ≈ TensorMap(ComplexF64[3, 0], physical, one(physical))
        @test info.relative_error <= 1.0e-12
        @test isfinite(norm(solution))

        zero_rhs = zero(supported)
        zero_solution, zero_info = SolverCE._solve_cluster_equation(action, zero_rhs)
        @test iszero(norm(zero_solution))
        @test iszero(zero_info.relative_error)
        @test zero_info.numiter == 0

        null_rhs = TensorMap(ComplexF64[0, 1], physical, one(physical))
        null_solution, null_info = SolverCE._solve_cluster_equation(action, null_rhs)
        @test iszero(norm(null_solution))
        @test null_info.relative_error == 1
        @test null_info.numiter == 0
        @test isfinite(norm(null_solution))

        zero_matrix = zero(matrix)
        zero_action = (tensor, flag) -> flag === Val(true) ? zero_matrix' * tensor : zero_matrix * tensor
        stationary, stationary_info = SolverCE._solve_cluster_equation(zero_action, supported)
        @test iszero(norm(stationary))
        @test stationary_info.relative_error == 1
        @test stationary_info.numiter == 0
    end

    @testset "Asymmetric U1 spaces in every bond direction" begin
        physical = U1Space(0 => 1, 1 => 1)
        unit = oneunit(physical)
        bond = U1Space(0 => 1, 1 => 1)
        spaces = level -> iszero(level) ? unit : bond
        for (direction, displacement) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1)))
            opposite = mod1(direction + 2, 4)
            sites = [(0, 0), displacement]
            levels = [ntuple(index -> index == direction ? 1 : 0, 4), ntuple(index -> index == opposite ? 1 : 0, 4)]
            factors = [solver_random_factor(rng, physical, level, spaces) for level in levels]
            rhs = solver_reference_contract(factors, sites)
            @test norm(rhs) > 0
            solved = SolverCE.solve_index(nothing, rhs, [1, 2], levels, (direction, opposite), spaces)
            @test length(solved) == 2
            for site in 1:2, index in 1:4
                expected = index > 2 ? spaces(levels[site][index])' : spaces(levels[site][index])
                @test domain(solved[site])[index] == expected
            end
            reconstructed = solver_reference_contract(solved, sites)
            @test norm(reconstructed - rhs) <= 1.0e-11 * norm(rhs)
        end
    end

    @testset "Environment adjoints with and without fermionic grading" begin
        for physical in (ℂ^2, Vect[FermionParity](0 => 1, 1 => 1))
            solver_check_adjoint(rng, physical, physical, [(0, 0), (1, 0), (2, 0)], [2])
            solver_check_adjoint(rng, physical, physical, [(0, 0), (1, 0), (2, 0), (3, 0)], [2, 3])
            solver_check_adjoint(rng, physical, physical, [(0, 0), (0, 1), (0, 2), (0, 3)], [2, 3])
        end
    end

    @testset "Plaquette SVD, zero residual, and improving sweeps" begin
        sites = [(0, 0), (0, 1), (1, 0), (1, 1)]
        cluster = SolverCE.Cluster(sites)
        physical = ℂ^2
        physical_product = ⊗(ntuple(_ -> physical, 4)...)
        rhs = randn(rng, ComplexF64, physical_product, physical_product)
        full_spaces = level -> iszero(level) ? ℂ^1 : ℂ^16
        exact, exact_error = SolverCE.solve_4_loop(rhs, cluster, full_spaces; maxsweeps = 0, verbosity = 0)
        exact_contraction = solver_reference_contract(exact, sites)
        @test norm(exact_contraction - rhs) <= 1.0e-11 * norm(rhs)
        @test exact_error <= 1.0e-11

        small_spaces = level -> iszero(level) ? ℂ^1 : ℂ^2
        zero_factors, zero_error = SolverCE.solve_4_loop(zero(rhs), cluster, small_spaces; verbosity = 0)
        @test iszero(zero_error)
        @test all(iszero, norm.(zero_factors))
        @test iszero(norm(solver_reference_contract(zero_factors, sites)))

        _, initial_error = SolverCE.solve_4_loop(rhs, cluster, small_spaces; maxsweeps = 0, verbosity = 0)
        fitted, fitted_error = SolverCE.solve_4_loop(
            rhs, cluster, small_spaces;
            maxsweeps = 2, linsolve_maxiter = 30, tol = 1.0e-10, verbosity = 0, seed = 91
        )
        measured_error = norm(solver_reference_contract(fitted, sites) - rhs) / norm(rhs)
        @test all(isfinite, norm.(fitted))
        @test fitted_error <= initial_error + 1.0e-12
        @test isapprox(fitted_error, measured_error; atol = 1.0e-12, rtol = 1.0e-12)
    end
end
