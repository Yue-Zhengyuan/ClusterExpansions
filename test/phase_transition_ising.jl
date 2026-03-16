using Test
using TensorKit
using TensorKitTensors
using ClusterExpansions
using PEPSKit
using Random

Random.seed!(1654841489)

# Set up time evolution algorithm
Δβ = 0.1
maxiter = 10

# Set up PEPO truncation algorithm
Dcut = 5
trunc_alg = LocalApprox(truncrank(Dcut))

gs = [2.5, 0.0]
Tcs = [1.2737, 2 / (log(1 + sqrt(2)))]

function converge_ctmenv(ρ::InfinitePEPO)
    pf = InfinitePartitionFunction(ρ)
    env = CTMRGEnv(ones, ComplexF64, InfinitePartitionFunction(ρ), ℂ^1)
    for (χ, verbosity) in zip([1, 5, 15], [0, 0, 2])
        trunc = truncrank(χ)
        ctm_alg = SequentialCTMRG(; verbosity, trunc, projector_alg = :fullinfinite)
        env, = leading_boundary(env, pf, ctm_alg)
    end
    return env
end

@testset "Transverse field Ising model (g = $(g))" for (g, Tc) in zip(gs, Tcs)
    J, z = 1.0, 0.0
    ce_alg = ising_operators(J, g, z; T = ComplexF64)
    βc = 1 / Tc # Critical temperature

    # get PEPO for exp(-H Δβ)
    ρ0 = InfinitePEPO(evolution_operator(ce_alg, Δβ))
    ρ = deepcopy(ρ0)

    # observables
    σx = SpinOperators.σˣ()
    σz = SpinOperators.σᶻ()
    lattice = fill(space(σx, 1), (1, 1))
    σx = LocalOperator(lattice, (CartesianIndex(1, 1),) => σx)
    σz = LocalOperator(lattice, (CartesianIndex(1, 1),) => σz)

    # Perform the time evolution
    β = Δβ
    for iter in 1:maxiter
        D0s = [space(t) for t in ρ0.A]
        Ds = [space(t) for t in ρ.A]
        env = converge_ctmenv(ρ)
        mx = expectation_value(ρ, σx, env)
        mz = expectation_value(ρ, σz, env)

        D = domain(ρ[1, 1, 1], 1)
        @info "Virtual bond ρ = $(D)"
        @info "β = $(round(β; digits = 4))"
        @info "⟨σˣ⟩ = $(mx)"
        @info "⟨σᶻ⟩ = $(mz)"

        if abs(β - βc) > 2.0e-2
            @test (β < βc) ? (abs(mz) < 0.5) : (abs(mz) > 0.5)
        end
        if g == 0
            @test abs(mx) < 1.0e-6
        end
        if iter < maxiter
            ρ = approximate(ρ0, ρ, trunc_alg)
            normalize!.(ρ.A)
            β += Δβ
        end
    end
end
