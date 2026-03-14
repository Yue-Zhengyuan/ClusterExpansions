abstract type TimeEvolution end

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

function evolution_operator(ce_alg::ClusterExpansion, β::Number; T_conv = ComplexF64)
    if β == 0.0
        pspace = domain(ce_alg.onesite_op)[1]
        vspace = ce_alg.spaces(0)
        t = id(T_conv, pspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1, 4), (5, 6, 2, 3)))
    end
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O = zeros(T_conv, codomain(O_clust_full), domain(O_clust_full))
    for (f_full, f_conv) in zip(blocks(O_clust_full), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O # Don't normalize, otherwise Atsushi will be mad.
end

function MPSKit.time_evolve(
        ce_alg::ClusterExpansion,
        time_alg::StaticTimeEvolution,
        trunc_alg::LocalApprox,
        observable;
        finalize! = nothing,
        A0 = nothing,
        skip_first::Bool = false,
        initial_guesses = i -> nothing,
        saving::Bool = true,
        normalizing::Bool = true
    )
    As = AbstractTensorMap[evolution_operator(ce_alg, β) for β in time_alg.βs_helper]
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator(ce_alg, time_alg.β₀)
    else
        A = A0
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
            op1, op2 = InfinitePEPO(A), InfinitePEPO(As[ind])
            A = only(approximate(op1, op2, trunc_alg).A)
        elseif ind == i
            op = InfinitePEPO(A)
            A = only(approximate(op, op, trunc_alg).A)
        else
            @error "Cannot perform time evolution without saving intermediaire steps for this time algorithm"
        end
        if normalizing
            A /= norm(A)
        end
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
