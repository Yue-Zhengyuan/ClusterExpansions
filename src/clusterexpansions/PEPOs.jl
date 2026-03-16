function init_PEPO(T, β, trivspace::ElementarySpace, onesite_op::AbstractTensorMap)
    exp_H = exp(-β * onesite_op)
    exp_H_perm = permute(exp_H, ((1, 2), ()))
    Isom = permute(isomorphism(T, trivspace' ⊗ trivspace', trivspace' ⊗ trivspace'), ((), (1, 2, 3, 4)))
    A = exp_H_perm * Isom
    return Dict((0, 0, 0, 0) => A)
end

function init_PEPO(T, β, onesite_op::AbstractTensorMap, trivspace)
    I = sectortype(onesite_op)
    return init_PEPO(T, β, trivspace, onesite_op)
end

function get_sum_space(highest, highest_loop, spaces)
    summedspace = ⊞([spaces(i) for i in 0:highest]...)
    # zerospace = ℂ^0
    # total_space = zerospace
    for ind in -1:-1:highest_loop
        summedspace = summedspace ⊞ spaces(ind)
    end
    return summedspace
end

function get_PEPO(T, pspace, PEPO, spaces)
    highest = [maximum([i[dir] for i in keys(PEPO)]) for dir in 1:4]
    highest_loop = [minimum([i[dir] for i in keys(PEPO)]) for dir in 1:4]
    conjugated = Bool[0, 0, 1, 1]
    Oout = SumSpace(pspace) ⊗ SumSpace(pspace)'
    Oin = ⊗(
        [
            conj ? (get_sum_space(h, hloop, spaces))' : get_sum_space(h, hloop, spaces)
                for (conj, h, hloop) in zip(conjugated, highest, highest_loop)
        ]...
    )
    O = zeros(T, Oout ← Oin)
    for (key, tens) in PEPO
        key = [i < 0 ? h - i + 1 : i + 1 for (i, h) in zip(key, highest)]
        # places = [get_location_PEPO(ind, highest[dir], spaces) for (dir,ind) = enumerate(key)]
        O[1, 1, key...] = tens
        # O[][:,:,places[1],places[2],places[3],places[4]] = tens[]
    end
    return O
end

function evolution_operator(ce_alg::ClusterExpansion, β::Number; T_conv = ComplexF64)
    if β == 0.0
        pspace = domain(ce_alg.onesite_op)[1]
        vspace = ce_alg.spaces(0)
        t = id(T_conv, pspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1, 4), (5, 6, 2, 3)))
    end
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O = zeros(T_conv, codomain(O_clust_full), domain(O_clust_full))
    for (f_full, f_conv) in zip(blocks(O_clust_full), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O # Don't normalize, otherwise Atsushi will be mad.
end
