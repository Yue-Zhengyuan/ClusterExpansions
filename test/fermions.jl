using Test
using LinearAlgebra
using TensorKit
using ClusterExpansions

@testset "Fermionic finite-cluster exponentials" begin
    hopping, interaction, chemical, offset, pairing = 0.41, 0.27, 0.19, 0.03, 0.07
    number = ComplexF64[0 0; 0 1]
    shifted_number = number - Matrix{ComplexF64}(I, 2, 2) / 2
    h1 = (2 * interaction * offset - chemical) * number
    h2 = interaction * kron(shifted_number, shifted_number)
    h2[2, 3] = h2[3, 2] = -hopping
    h2[1, 4] = h2[4, 1] = pairing
    alg = spinless_fermion_operators(hopping, interaction, chemical; b = offset, δ = pairing, p = 3)
    @test local_matrix(alg.onesite_op) ≈ h1
    @test local_matrix(alg.twosite_op) ≈ h2

    # Each even fermionic bond acts on consecutive modes in this Fock ordering.
    # Its dense embedding therefore needs no Jordan-Wigner string outside the bond.
    geometries = (
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 1)],
    )
    output = evolution_operator(alg, 0.13)
    for sites in geometries
        H = dense_cluster_hamiltonian(h2, h1, sites)
        @test finite_pepo_matrix(output, sites) ≈ exp(-0.13 * H) rtol = 5.0e-11 atol = 5.0e-12
    end

    @testset "Composition across fermionic virtual bonds" begin
        second_step = 0.04
        second_layer = evolution_operator(alg, second_step)
        combined = combine_layers(output, second_layer)
        for sites in geometries[1:2]
            H = dense_cluster_hamiltonian(h2, h1, sites)
            first_matrix = finite_pepo_matrix(output, sites)
            second_matrix = finite_pepo_matrix(second_layer, sites)
            combined_matrix = finite_pepo_matrix(combined, sites)
            @test second_matrix ≈ exp(-second_step * H) rtol = 5.0e-11 atol = 5.0e-12
            @test combined_matrix ≈ second_matrix * first_matrix rtol = 5.0e-11 atol = 5.0e-12
            @test combined_matrix ≈ exp(-(0.13 + second_step) * H) rtol = 5.0e-11 atol = 5.0e-12
        end
    end

    sites = geometries[3]
    H = dense_cluster_hamiltonian(h2, h1, sites)
    realtime = finite_pepo_matrix(evolution_operator(alg, 0.07im), sites)
    @test realtime ≈ exp(-0.07im * H) rtol = 5.0e-11 atol = 5.0e-12
    @test realtime' * realtime ≈ Matrix{ComplexF64}(I, 8, 8) rtol = 5.0e-11 atol = 5.0e-12
end
