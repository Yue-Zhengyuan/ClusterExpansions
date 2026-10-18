using LinearAlgebra
using TensorKit

"""Convert a physical tensor operator to an ordinary matrix with site one varying fastest."""
function local_matrix(operator::AbstractTensorMap)
    return reshape(convert(Array, operator), dim(codomain(operator)), dim(domain(operator)))
end

"""Assemble a small bosonic Hamiltonian by explicit basis states and local matrix entries."""
function dense_cluster_hamiltonian(h2::AbstractMatrix, h1::AbstractMatrix, sites::AbstractVector)
    d = size(h1, 1)
    N = length(sites)
    size(h1) == (d, d) || throw(DimensionMismatch("The site operator must be square"))
    size(h2) == (d^2, d^2) || throw(DimensionMismatch("The bond operator must act on two sites"))
    basis = [digits(index; base = d, pad = N) .+ 1 for index in 0:(d^N - 1)]
    bonds = [(i, j) for i in 1:N for j in (i + 1):N if sum(abs, sites[i] .- sites[j]) == 1]
    H = zeros(promote_type(eltype(h1), eltype(h2)), d^N, d^N)
    for (row, bra) in enumerate(basis), (column, ket) in enumerate(basis)
        for site in 1:N
            all(other -> other == site || bra[other] == ket[other], 1:N) || continue
            H[row, column] += h1[bra[site], ket[site]]
        end
        for (i, j) in bonds
            all(other -> other in (i, j) || bra[other] == ket[other], 1:N) || continue
            H[row, column] += h2[bra[i] + d * (bra[j] - 1), ket[i] + d * (ket[j] - 1)]
        end
    end
    return H
end

"""Contract public PEPO tensors on a finite patch using independent lattice wiring and empty boundaries."""
function finite_pepo_matrix(tensor::AbstractTensorMap, sites::AbstractVector)
    tensor = convert(TensorMap, tensor)
    N = length(sites)
    locations = Dict(site => i for (i, site) in enumerate(sites))
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = (3, 4, 1, 2)
    edges = Tuple{Int, Int, Int}[]
    for (i, site) in enumerate(sites), direction in 1:4
        step = directions[direction]
        neighbor = (site[1] + step[1], site[2] + step[2])
        j = get(locations, neighbor, 0)
        i < j && push!(edges, (i, j, direction))
    end
    boundary_count = 4 * N - 2 * length(edges)
    wiring = [[-site, -N - site, 0, 0, 0, 0] for site in 1:N]
    for (edge, (i, j, direction)) in enumerate(edges)
        wiring[i][direction + 2] = boundary_count + edge
        wiring[j][opposites[direction] + 2] = boundary_count + edge
    end
    tensors = AbstractTensorMap[fill(tensor, N)...]
    label = 0
    for site in 1:N, direction in 1:4
        iszero(wiring[site][direction + 2]) || continue
        label += 1
        wiring[site][direction + 2] = label
        virtual = domain(tensor)[direction]
        unit = oneunit(virtual)
        boundary = isometry(scalartype(tensor), virtual, unit) * ones(scalartype(tensor), unit)
        push!(tensors, boundary)
        push!(wiring, [label])
    end
    result = ncon(tensors, wiring)
    return local_matrix(permute(result, (Tuple(1:N), Tuple((N + 1):(2 * N)))))
end
