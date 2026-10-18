using Test
using ClusterExpansions
using PEPSKit
using LinearAlgebra

@testset "Constructed layers on finite patches" begin
    sites = [(0, 0), (0, 1)]
    alg = ising_operators(0.7, 0.0, 0.2; p = 2)
    H = dense_cluster_hamiltonian(local_matrix(alg.twosite_op), local_matrix(alg.onesite_op), sites)
    first_step, second_step = 0.09, 0.04
    first_layer = evolution_operator(alg, first_step)
    second_layer = evolution_operator(alg, second_step)
    combined = combine_layers(first_layer, second_layer)
    expected = exp(-(first_step + second_step) * H)
    @test finite_pepo_matrix(combined, sites) ≈ expected rtol = 2.0e-11 atol = 2.0e-12
    repeated = combine_layers(first_layer, 2; method = :squaring)
    @test finite_pepo_matrix(repeated, sites) ≈ exp(-2 * first_step * H) rtol = 2.0e-11 atol = 2.0e-12
    cell = combine_layers(InfinitePEPO(first_layer), InfinitePEPO(second_layer))
    @test finite_pepo_matrix(cell[1, 1, 1], sites) ≈ expected rtol = 2.0e-11 atol = 2.0e-12
end

@testset "Noncommuting constructed quantum layers" begin
    sites = [(0, 0), (0, 1)]
    identity = Matrix{ComplexF64}(I, 2, 2)
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    transverse = kron(σx, identity) + kron(identity, σx)
    longitudinal = kron(σz, identity) + kron(identity, σz)
    interaction = kron(σz, σz)
    first_J, first_g, first_z = 0.7, 0.43, 0.19
    second_J, second_g, second_z = 0.4, -0.31, -0.23
    first_H = -first_J * (interaction + first_g * transverse + first_z * longitudinal)
    second_H = -second_J * (interaction + second_g * transverse + second_z * longitudinal)
    @test norm(first_H * second_H - second_H * first_H) > 0.1

    first_alg = ising_operators(first_J, first_g, first_z; p = 2)
    second_alg = ising_operators(second_J, second_g, second_z; p = 2)
    first_step, second_step = 0.09, 0.07
    first_exp = exp(-first_step * first_H)
    second_exp = exp(-second_step * second_H)
    expected = second_exp * first_exp
    @test norm(expected - first_exp * second_exp) > 1.0e-5

    first_layer = evolution_operator(first_alg, first_step)
    second_layer = evolution_operator(second_alg, second_step)
    combined = combine_layers(first_layer, second_layer)
    @test finite_pepo_matrix(combined, sites) ≈ expected rtol = 2.0e-11 atol = 2.0e-12
end
