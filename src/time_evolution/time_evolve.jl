abstract type TimeEvolution end

# struct StaticTimeEvolution <: TimeEvolution
#     β₀
#     Δβ
#     maxiter
#     trunc_alg
#     verbosity
# end

struct TimeDependentTimeEvolution <: TimeEvolution
    β₀
    Δβ
    maxiter
    verbosity
    f₁
    f₂
end

struct StaticTimeEvolution <: TimeEvolution
    β₀
    βs_helper
    update_list
    verbosity
end

function StaticTimeEvolution(β₀, βs_helper, update_list; verbosity = 0)
    return StaticTimeEvolution(β₀, βs_helper, update_list, verbosity)
end

function UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 0)
    return StaticTimeEvolution(β₀, [Δβ], [1 for i in 1:maxiter], verbosity)
end

function SquaringTimeEvolution(β₀, maxiter; verbosity = 0)
    return StaticTimeEvolution(β₀, [], 1:maxiter, verbosity)
end

function TimeDependentTimeEvolution(β₀, Δβ, maxiter; verbosity = 0, f₁ = β -> 1.0, f₂ = β -> 1.0)
    return TimeDependentTimeEvolution(β₀, Δβ, maxiter, verbosity, f₁, f₂)
end

function evolution_operator(ce_alg::ClusterExpansion, time_alg::TimeDependentTimeEvolution, β::Number; T_conv = ComplexF64, canoc_alg::Union{Nothing, Canonicalization} = nothing)
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, time_alg.Δβ, time_alg.f₂(β) * ce_alg.twosite_op, time_alg.f₁(β) * ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O
end

function evolution_operator(ce_alg::ClusterExpansion, β::Number; T_conv = ComplexF64, canoc_alg::Union{Nothing, Canonicalization} = nothing)
    if β == 0.0
        pspace = domain(ce_alg.onesite_op)[1]
        vspace = ce_alg.spaces(0)
        t = id(T_conv, pspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1, 4), (5, 6, 2, 3)))
    end
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O # Don't normalize, otherwise Atsushi will be mad.
end

function MPSKit.time_evolve(
        ce_alg::ClusterExpansion,
        time_alg::StaticTimeEvolution,
        trunc_alg::EnvTruncation,
        observable;
        finalize! = nothing,
        A0 = nothing,
        canoc_alg::Union{Canonicalization, Nothing} = nothing,
        skip_first::Bool = false,
        initial_guesses = i -> nothing,
        saving::Bool = true,
        normalizing::Bool = true
    )
    As = AbstractTensorMap[evolution_operator(ce_alg, β; canoc_alg) for β in time_alg.βs_helper]
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator(ce_alg, time_alg.β₀; canoc_alg)
    else
        A = canonicalize(A0, canoc_alg)
    end
    if skip_first
        obs = nothing
        expvals = []
    else
        obs = observable(A, 0)
        expvals = [obs]
    end
    push!(As, copy(A))
    push!(times, time_alg.β₀)

    for (i, ind) in enumerate(time_alg.update_list)
        if ind <= length(As)
            A, _ = approximate_state((A, As[ind]), trunc_alg)
        elseif ind == i
            A, _ = approximate_state((A, A), trunc_alg)
        else
            @error "Cannot perform time evolution without saving intermediaire steps for this time algorithm"
        end
        if normalizing
            A /= norm(A)
        end
        A = canonicalize(A, canoc_alg)
        obs = observable(A, i)
        push!(times, times[end] + times[ind])
        if saving
            push!(expvals, obs)
            push!(As, copy(A))
        end
        if time_alg.verbosity > 1
            @info "Time evolution step $(i) with β = $(times[end]), obs = $(obs)"
            @info "Bond dimension is now $(dim(domain(A)[1]))"
            if time_alg.verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if !isnothing(finalize!)
            finalize!(As, expvals, i)
        end
    end
    if saving
        return times[(length(time_alg.βs_helper) + 1):end], expvals, As[(length(time_alg.βs_helper) + 1):end]
    else
        return times[end], obs, A
    end
end

function get_time_array(time_alg::StaticTimeEvolution)
    times = copy(time_alg.βs_helper)
    push!(times, time_alg.β₀)
    for ind in time_alg.update_list
        push!(times, times[end] + times[ind])
    end
    return times[(length(time_alg.βs_helper) + 1):end]
end


function time_scan(
        ce_alg::ClusterExpansion,
        times::Array,
        observable;
        verbosity::Int = 0,
        finalize! = nothing
    )
    expvals = []
    As = []
    for (i, t) in enumerate(times)
        A = evolution_operator(ce_alg, t)
        obs = observable(A)

        push!(expvals, obs)
        push!(As, copy(A))

        if verbosity > 1
            @info "Time evolution step $(i) with β = $(t), obs = $(obs)"
            if verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if !isnothing(finalize!)
            finalize!(A, obs, i)
        end
    end
    return times, expvals, As
end
