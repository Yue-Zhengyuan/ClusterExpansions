using Test
using LinearAlgebra
using TensorKit
using ClusterExpansions

"""Build ungraded local operators from explicitly specified dense matrices."""
function construction_operators()
    pspace = ℂ^2
    identity = Matrix{ComplexF64}(I, 2, 2)
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    h1 = 0.31 * identity + 0.23 * σx - 0.17 * σz
    # The unequal factors also expose accidental reversals of a bond's site order.
    h2 = -0.43 * kron(σz, σz) + 0.09 * kron(σx, σz)
    onsite = TensorMap(h1, pspace ← pspace)
    bond = TensorMap(reshape(h2, 2, 2, 2, 2), pspace ⊗ pspace ← pspace ⊗ pspace)
    return bond, onsite, h2, h1
end

@testset "On-site evolution preserves scale" begin
    bond, onsite, _, h1 = construction_operators()
    alg = ClusterExpansion(zero(bond), onsite; p = 1)
    step = 0.27
    blocks, tensor = clusterexpansion(alg, step)
    @test Set(keys(blocks)) == Set([(0, 0, 0, 0)])
    @test finite_pepo_matrix(tensor, [(0, 0)]) ≈ exp(-step * h1) atol = 2.0e-13 rtol = 2.0e-13

    sites = [(0, 0), (0, 1), (1, 1)]
    expected = exp(-step * dense_cluster_hamiltonian(zeros(ComplexF64, 4, 4), h1, sites))
    output = evolution_operator(alg, step)
    @test finite_pepo_matrix(output, sites) ≈ expected atol = 5.0e-13 rtol = 5.0e-13
    output32 = evolution_operator(alg, step; T_conv = ComplexF32)
    @test scalartype(output32) == ComplexF32
    @test finite_pepo_matrix(output32, sites) ≈ expected atol = 3.0e-7 rtol = 3.0e-7
end

@testset "Zero time is the identity with interactions present" begin
    bond, onsite, _, _ = construction_operators()
    alg = ClusterExpansion(bond, onsite; p = 3)
    blocks, tensor = clusterexpansion(alg, 0)
    @test Set(keys(blocks)) == Set([(0, 0, 0, 0)])
    for sites in ([(0, 0)], [(0, 0), (0, 1)], [(0, 0), (0, 1), (1, 1)])
        actual = finite_pepo_matrix(tensor, sites)
        @test actual ≈ Matrix{ComplexF64}(I, 2^length(sites), 2^length(sites)) atol = 2.0e-13 rtol = 2.0e-13
    end
    @test finite_pepo_matrix(evolution_operator(alg, 0), [(0, 0)]) ≈ Matrix{ComplexF64}(I, 2, 2)
end

@testset "Two-site exponentials retain bond orientation" begin
    bond, onsite, h2, h1 = construction_operators()
    alg = ClusterExpansion(bond, onsite; p = 2)
    _, tensor = clusterexpansion(alg, 0.14)
    @test length(alg.clusters[1]) == 2
    for sites in ([(0, 0), (0, 1)], [(0, 0), (1, 0)])
        expected = exp(-0.14 * dense_cluster_hamiltonian(h2, h1, sites))
        @test finite_pepo_matrix(tensor, sites) ≈ expected atol = 5.0e-12 rtol = 5.0e-12
    end
end

@testset "Three-site chain and corner exponentials" begin
    bond, onsite, h2, h1 = construction_operators()
    alg = ClusterExpansion(bond, onsite; p = 3)
    @test length(alg.clusters[2]) == 6
    output = evolution_operator(alg, 0.12)
    geometries = (
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
    )
    for sites in geometries
        expected = exp(-0.12 * dense_cluster_hamiltonian(h2, h1, sites))
        @test finite_pepo_matrix(output, sites) ≈ expected atol = 2.0e-11 rtol = 2.0e-11
    end

    step = 0.16im
    _, realtime = clusterexpansion(alg, step)
    sites = geometries[3]
    actual = finite_pepo_matrix(realtime, sites)
    expected = exp(-step * dense_cluster_hamiltonian(h2, h1, sites))
    @test actual ≈ expected atol = 2.0e-11 rtol = 2.0e-11
    @test actual' * actual ≈ Matrix{ComplexF64}(I, 8, 8) atol = 3.0e-11 rtol = 3.0e-11
end

@testset "Four-site paths and branching trees" begin
    bond, onsite, h2, h1 = construction_operators()
    alg = ClusterExpansion(bond, onsite; p = 4, solving_loops = false)
    @test length(alg.clusters[3]) == 18
    @test all(cluster -> iszero(cluster.m), alg.clusters[3])
    _, tensor = clusterexpansion(alg, 0.13)
    # All orientations matter: a bend must not seed incompatible gauges for a later straight path.
    # Paths require a two-site environment solve; branches require a one-site solve.
    for cluster in alg.clusters[3]
        sites = cluster.cluster
        @testset "Sites $sites" begin
            expected = exp(-0.13 * dense_cluster_hamiltonian(h2, h1, sites))
            @test finite_pepo_matrix(tensor, sites) ≈ expected atol = 8.0e-11 rtol = 8.0e-11
        end
    end
end

@testset "Symmetry-graded finite-cluster exponentials" begin
    models = (
        ("Z2 Ising", ising_operators(0.61, 0.27, 0.0; spin_symmetry = Z2Irrep, p = 3)),
        ("U1 Heisenberg", heisenberg_operators(0.38, 0.38, -0.21, 0.17; spin_symmetry = U1Irrep, p = 3)),
        ("SU2 Heisenberg", heisenberg_operators(0.42, 0.42, 0.42, 0.0; spin_symmetry = SU2Irrep, p = 3)),
    )
    for (name, alg) in models
        @testset "$name" begin
            h2, h1 = local_matrix(alg.twosite_op), local_matrix(alg.onesite_op)
            _, tensor = clusterexpansion(alg, 0.15)
            for sites in ([(0, 0), (0, 1), (0, 2)], [(0, 0), (0, 1), (1, 1)])
                expected = exp(-0.15 * dense_cluster_hamiltonian(h2, h1, sites))
                @test finite_pepo_matrix(tensor, sites) ≈ expected atol = 5.0e-11 rtol = 5.0e-11
            end
        end
    end
end

@testset "Asymmetric charge spaces preserve virtual-leg duals" begin
    virtual = U1Space(0 => 1, 1 => 1)
    reference = heisenberg_operators(0.4, 0.4, 0.2, 0.0; spin_symmetry = U1Irrep, p = 2)
    unit = oneunit(domain(reference.onesite_op)[1])
    spaces = level -> iszero(level) ? unit : virtual
    alg = ClusterExpansion(reference.twosite_op, reference.onesite_op; p = 2, spaces)
    blocks, _ = clusterexpansion(alg, 0.17)
    for direction in 1:4
        level = ntuple(index -> index == direction ? 1 : 0, 4)
        @test haskey(blocks, level)
        expected = direction > 2 ? virtual' : virtual
        @test domain(blocks[level])[direction] == expected
    end
end

@testset "Classical plaquette exponential" begin
    alg = ising_operators(0.63, 0.0, 0.19; p = 4, loop_space = ℂ^4, loop_maxsweeps = 0)
    @test length(alg.clusters[3]) == 19
    blocks, tensor = clusterexpansion(alg, 0.11)
    @test count(key -> any(level -> level < 0, key), keys(blocks)) == 4
    sites = [(0, 0), (0, 1), (1, 0), (1, 1)]
    h2, h1 = local_matrix(alg.twosite_op), local_matrix(alg.onesite_op)
    expected = exp(-0.11 * dense_cluster_hamiltonian(h2, h1, sites))
    @test finite_pepo_matrix(tensor, sites) ≈ expected atol = 8.0e-11 rtol = 8.0e-11
end

@testset "Noncommuting plaquette correction by full-rank SVD" begin
    bond, onsite, h2, h1 = construction_operators()
    spaces = level -> level < 0 ? ℂ^16 : ℂ^(4^level)
    alg = ClusterExpansion(bond, onsite; p = 4, spaces, loop_maxsweeps = 0)
    @test length(alg.clusters[3]) == 19
    step = 0.24
    blocks, tensor = clusterexpansion(alg, step)
    loop_blocks = [block for (levels, block) in blocks if any(level -> level < 0, levels)]
    @test length(loop_blocks) == 4
    @test all(block -> norm(block) > 0, loop_blocks)
    sites = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected = exp(-step * dense_cluster_hamiltonian(h2, h1, sites))
    @test finite_pepo_matrix(tensor, sites) ≈ expected atol = 8.0e-11 rtol = 8.0e-11
end
