struct PEPOObservable
    obs::Union{LocalOperator, InfinitePEPO, TensorMap, Function, Symbol}
    env_alg::Union{PEPSKit.CTMRGAlgorithm, VUMPS, Nothing}
    func::Function
end

function PEPO_observables(obss::Vector, env_alg::Union{PEPSKit.CTMRGAlgorithm, VUMPS, Nothing}; func::Function = x -> x)
    return [PEPOObservable(obs, env_alg, func) for obs in obss]
end

function MPSKit.expectation_value(ρ::InfinitePEPO, obs::TensorMap{T, S, 1, 1}, (mps, env)::Tuple) where {T, S}
    O = ρ[1, 1]
    E_num = PEPSKit.@autoopt @tensor twist(O, 2)[d1 d2; DN DE DS DW] * mps.AC[1][DtL DN; DtR] *
        conj(mps.AC[1][DbL DS; DbR]) * obs[d2; d1] *
        env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    E_denom = PEPSKit.@autoopt @tensor twist(O, 2)[d d; DN DE DS DW] * mps.AC[1][DtL DN; DtR] *
        conj(mps.AC[1][DbL DS; DbR]) *
        env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    return E_num / E_denom
end

function MPSKit.expectation_value(::InfinitePEPO, symb::Symbol, (mps, env)::Tuple)
    if symb == :spectrum
        ϵ, δ, θ = marek_gap(mps; num_vals = 20)
        return 1 / ϵ, δ, θ
    else
        @warn "Observable $(symb) not defined. This will be set to zero"
        return 0
    end
end

function _env_algs(observables::Vector{PEPOObservable})
    return [obs.env_alg isa VUMPS ? :VUMPS : (obs.env_alg isa PEPSKit.CTMRGAlgorithm ? :CTMRG : :nothing) for obs in observables]
end

function get_env_alg(observables::Vector{PEPOObservable}, alg_type)
    algs = [obs.env_alg for obs in observables if obs.env_alg isa alg_type]
    if !all([algs[1] == alg for alg in algs])
        @warn "Algorithms are different. Using only the first one"
    end
    if algs == []
        return nothing
    end
    return algs[1]
end

function trace_physicalspaces(O::PEPSKit.PEPOTensor)
    @plansor t[W S; N E] := O[p p; N E S W]
    return t
end

function trace_out(ρ::InfinitePEPO)
    @assert size(ρ, 3) == 1
    return InfinitePartitionFunction(
        trace_physicalspaces.(reshape(ρ.A, size(ρ, 1), size(ρ, 2)))
    )
end

function calculate_observables(O::AbstractTensorMap{E, S, 2, 4}, χ::Int, observables) where {E, S}
    envspace = _envspace(codomain(O)[1])(χ)
    env_algs = _env_algs(observables)

    vumps_alg = get_env_alg(observables, VUMPS)
    ctm_alg = get_env_alg(observables, PEPSKit.CTMRGAlgorithm)
    ρ = InfinitePEPO(O)
    pf = trace_out(ρ)
    if :VUMPS ∈ env_algs
        T = InfiniteMPO([pf[1, 1]])
        pspace = domain(pf[1, 1])[2]

        mps = InfiniteMPS(
            [
                randn(
                    ComplexF64,
                    envspace * pspace,
                    envspace,
                ),
            ]
        )
        mps, env, _ = leading_boundary(mps, T, vumps_alg)
        vumps_env = (mps, env)
    end
    if :CTMRG ∈ env_algs
        ctmrg_env, = leading_boundary(CTMRGEnv(pf, envspace), pf, ctm_alg)
    end
    return [env_type == :VUMPS ? expectation_value(obs.func(ρ), obs.obs, vumps_env) : (env_type == :CTMRG ? expectation_value(obs.func(ρ), obs.obs, ctmrg_env) : obs.obs(obs.func(ρ))) for (obs, env_type) in zip(observables, env_algs)]
end
