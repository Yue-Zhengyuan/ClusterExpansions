"""Embed a local operator into the ordered physical sites of a finite cluster."""
function _embed_cluster_operator(operator::AbstractTensorMap, sites::Tuple, N::Integer, identity::AbstractTensorMap)
    remaining = [site for site in 1:N if site ∉ sites]
    tensors = [operator, fill(identity, length(remaining))...]
    indices = [vcat([-site for site in sites], [-N - site for site in sites])]
    append!(indices, [[-site, -N - site] for site in remaining])
    term = ncon(tensors, indices)
    return permute(term, (Tuple(1:N), Tuple((N + 1):(2 * N))))
end

"""Accumulate local Hamiltonian terms in one buffer and exponentiate the cluster."""
function exponentiate_hamiltonian(twosite_op::AbstractTensorMap, onesite_op::AbstractTensorMap, cluster::Cluster, β::Number)
    N = cluster.N
    pspace = domain(onesite_op)[1]
    T = promote_type(scalartype(twosite_op), scalartype(onesite_op))
    identity = id(T, pspace)
    physical = ⊗(ntuple(_ -> pspace, N)...)
    H = zeros(T, physical ← physical)
    if !iszero(norm(twosite_op))
        for bond in cluster.bonds_indices
            axpy!(one(T), _embed_cluster_operator(twosite_op, bond, N, identity), H)
        end
    end
    if !iszero(norm(onesite_op))
        for site in 1:N
            axpy!(one(T), _embed_cluster_operator(onesite_op, (site,), N, identity), H)
        end
    end
    return exp(-β * H)
end

"""Contract the currently solved PEPO on a finite cluster with empty boundary legs."""
function contract_PEPO(::Type{T}, cluster::Cluster, PEPO::AbstractDict, spaces) where {T}
    N = cluster.N
    indices = zeros(Int, N, 6)
    for site in 1:N
        indices[site, 1] = -site
        indices[site, 2] = -N - site
    end
    boundary = trues(N, 4)
    # Contract dimension-one boundary vectors before the internal cluster bonds.
    boundary_count = 4 * N - 2 * length(cluster.bonds_indices)
    for (bond, (i, j)) in enumerate(cluster.bonds_indices)
        a, b = get_direction(cluster.cluster[i], cluster.cluster[j])
        indices[i, a + 2] = boundary_count + bond
        indices[j, b + 2] = boundary_count + bond
        boundary[i, a] = false
        boundary[j, b] = false
    end

    masks = [ntuple(dir -> boundary[site, dir], 4) for site in 1:N]
    bounds = _level_bounds(PEPO)
    pspace = codomain(PEPO[(0, 0, 0, 0)])[1]
    first_tensor = get_PEPO(T, pspace, PEPO, spaces; boundary = masks[1], level_bounds = bounds)
    cache = Dict(masks[1] => first_tensor)
    tensors = [
        get!(cache, mask) do
            get_PEPO(T, pspace, PEPO, spaces; boundary = mask, level_bounds = bounds)
        end for mask in masks
    ]

    # Restrict outward legs before assembling each direct-sum tensor, avoiding D^4
    # intermediate tensors at sites that only carry one or two internal bonds.
    unit = spaces(0)
    trivial = SumSpace(unit)
    vectors = [spzeros(T, dir > 2 ? trivial' : trivial) for dir in 1:4]
    for dir in 1:4
        vectors[dir][1] = ones(T, dir > 2 ? unit' : unit)
    end
    contractions = [collect(@view indices[site, :]) for site in 1:N]
    all_tensors = AbstractTensorMap[tensors...]
    label = 0
    for site in 1:N, dir in 1:4
        boundary[site, dir] || continue
        label += 1
        contractions[site][dir + 2] = label
        push!(all_tensors, vectors[dir])
        push!(contractions, [label])
    end
    result = ncon(all_tensors, contractions)
    # Each remaining physical SumSpace has one summand, so unwrap its sole block.
    return only(permute(result, (Tuple(1:N), Tuple((N + 1):(2 * N)))))
end
