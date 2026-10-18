using Test
using TensorKit
using PEPSKit
using ClusterExpansions
using LinearAlgebra
using Random

"""Represent a dense one-site matrix as a PEPO tensor with trivial virtual bonds."""
function _layer_test_product_operator(matrix::AbstractMatrix)
    d = size(matrix, 1)
    size(matrix, 2) == d || throw(ArgumentError("The reference matrix must be square"))
    physical, unit = ℂ^d, ℂ^1
    data = reshape(copy(matrix), d, d, 1, 1, 1, 1)
    return TensorMap(data, physical ⊗ physical' ← unit ⊗ unit ⊗ unit' ⊗ unit')
end

"""Extract the dense physical matrix of a PEPO whose virtual bonds are trivial."""
function _layer_test_physical_matrix(tensor::PEPSKit.PEPOTensor)
    @assert all(dim(space) == 1 for space in domain(tensor))
    return reshape(convert(Array, tensor), dim(codomain(tensor)[1]), dim(codomain(tensor)[2]))
end

"""Bend an operator's incoming physical index and append trivial PEPO virtual legs."""
function _layer_test_operator_tensor(operator::AbstractTensorMap)
    tensor = permute(operator, ((1, 2), ()))
    for direction in 1:4
        index = numout(tensor) + numin(tensor) + 1
        tensor = insertleftunit(tensor, Val(index); dual = direction > 2)
    end
    return tensor
end

"""Remove trivial virtual legs and restore the incoming physical index of an onsite PEPO."""
function _layer_test_onsite_operator(tensor::PEPSKit.PEPOTensor)
    for _ in 1:4
        tensor = removeunit(tensor, Val(3))
    end
    return permute(tensor, ((1,), (2,)))
end

"""Multiply physical matrix slices and combine virtual indices in ordinary array order."""
function _layer_test_dense_product(A::AbstractArray{TA, 6}, B::AbstractArray{TB, 6}) where {TA, TB}
    adims, bdims = size(A)[3:6], size(B)[3:6]
    fused_dims = ntuple(direction -> adims[direction] * bdims[direction], 4)
    result = zeros(promote_type(TA, TB), size(B, 1), size(A, 2), fused_dims...)
    for a in CartesianIndices(adims), b in CartesianIndices(bdims)
        indices = ntuple(direction -> a[direction] + adims[direction] * (b[direction] - 1), 4)
        result[:, :, indices...] .= view(B, :, :, Tuple(b)...) * view(A, :, :, Tuple(a)...)
    end
    return result
end

"""Construct a periodic two-by-two cell with site-dependent vertical and horizontal bonds."""
function _layer_test_random_cell(rng::AbstractRNG, north::AbstractMatrix{<:Integer}, east::AbstractMatrix{<:Integer})
    physical = ℂ^2
    tensors = [
        randn(
            rng, ComplexF64,
            physical ⊗ physical' ←
                ℂ^north[row, column] ⊗ ℂ^east[row, column] ⊗
                (ℂ^north[mod1(row + 1, 2), column])' ⊗ (ℂ^east[row, mod1(column - 1, 2)])'
        )
            for row in 1:2, column in 1:2, _ in 1:1
    ]
    return InfinitePEPO(tensors)
end

"""Contract a two-site periodic row directly, summing its two horizontal bonds."""
function _layer_test_periodic_row(tensor::PEPSKit.PEPOTensor)
    data = convert(Array, tensor)
    d, D = size(data, 1), size(data, 4)
    @assert size(data) == (d, d, 1, D, 1, D)
    result = zeros(eltype(data), d^2, d^2)
    for pout₁ in 1:d, pout₂ in 1:d, pin₁ in 1:d, pin₂ in 1:d, east in 1:D, west in 1:D
        row = pout₁ + d * (pout₂ - 1)
        column = pin₁ + d * (pin₂ - 1)
        result[row, column] += data[pout₁, pin₁, 1, east, 1, west] * data[pout₂, pin₂, 1, west, 1, east]
    end
    return result
end

@testset "PEPO layer composition" begin
    rng = Xoshiro(0x1ce0)
    α, β = 0.17, 0.23
    A_matrix = [cosh(α) -sinh(α); -sinh(α) cosh(α)]
    B_matrix = [exp(-β) 0.0; 0.0 exp(β)]
    A = _layer_test_product_operator(A_matrix)
    B = _layer_test_product_operator(B_matrix)
    expected = B_matrix * A_matrix

    @testset "Application order and scale" begin
        @test !isapprox(expected, A_matrix * B_matrix; atol = 1.0e-8)
        @test _layer_test_physical_matrix(apply_PEPO_exact(A, B)) ≈ expected
        combined, info = combine_layers(A, B; return_info = true)
        @test _layer_test_physical_matrix(combined) ≈ expected
        @test iszero(info.truncation_error)
        @test norm(combined) ≈ norm(expected)
        @test !isapprox(norm(combined), 1.0; atol = 1.0e-3)

        product_cell = combine_layers(InfinitePEPO(A), InfinitePEPO(B))
        @test size(product_cell) == (1, 1, 1)
        @test _layer_test_physical_matrix(product_cell[1, 1, 1]) ≈ expected

        mixed = combine_layers(
            _layer_test_product_operator(Float32.(A_matrix)),
            _layer_test_product_operator(ComplexF64.(B_matrix))
        )
        @test scalartype(mixed) == ComplexF64
        @test _layer_test_physical_matrix(mixed) ≈ ComplexF64.(B_matrix) * Float32.(A_matrix)
        mixed_compressed = combine_layers(
            _layer_test_product_operator(Float32.(A_matrix)),
            _layer_test_product_operator(ComplexF64.(B_matrix)); truncation = truncrank(1)
        )
        @test scalartype(mixed_compressed) == ComplexF64
        @test _layer_test_physical_matrix(mixed_compressed) ≈ ComplexF64.(B_matrix) * Float32.(A_matrix)
    end

    @testset "Application to PEPS" begin
        physical, unit = ℂ^2, ℂ^1
        vector = ComplexF64[1.0, 2im]
        state = TensorMap(reshape(copy(vector), 2, 1, 1, 1, 1), physical ← unit ⊗ unit ⊗ unit' ⊗ unit')
        @test vec(convert(Array, apply_PEPO_exact(state, B))) ≈ B_matrix * vector
        applied = apply_PEPO_exact(InfinitePEPS(state), InfinitePEPO(B))
        @test size(applied) == (1, 1)
        @test vec(convert(Array, applied[1, 1])) ≈ B_matrix * vector
    end

    @testset "Fermion-parity physical indices" begin
        physical = Vect[FermionParity](0 => 2, 1 => 2)
        bottom_operator = zeros(ComplexF64, physical ← physical)
        top_operator = zeros(ComplexF64, physical ← physical)
        block(bottom_operator, FermionParity(0)) .= A_matrix
        block(bottom_operator, FermionParity(1)) .= [cosh(2 * α) -sinh(2 * α); -sinh(2 * α) cosh(2 * α)]
        block(top_operator, FermionParity(0)) .= B_matrix
        block(top_operator, FermionParity(1)) .= [exp(-2 * β) 0.0; 0.0 exp(2 * β)]
        bottom = _layer_test_operator_tensor(bottom_operator)
        top = _layer_test_operator_tensor(top_operator)
        reference = top_operator * bottom_operator
        @test _layer_test_onsite_operator(bottom) ≈ bottom_operator
        @test !isapprox(reference, bottom_operator * top_operator; atol = 1.0e-8)
        @test _layer_test_onsite_operator(apply_PEPO_exact(bottom, top)) ≈ reference
        @test _layer_test_onsite_operator(combine_layers(bottom, top)) ≈ reference

        cell = combine_layers(InfinitePEPO(bottom), InfinitePEPO(top))
        @test _layer_test_onsite_operator(cell[1, 1, 1]) ≈ reference
        compressed, info = combine_layers(bottom, top; truncation = truncrank(1), return_info = true)
        @test _layer_test_onsite_operator(compressed) ≈ reference
        @test isfinite(info.truncation_error)
        @test info.truncation_error <= 1.0e-10
    end

    @testset "Anisotropic virtual bonds" begin
        physical, north, east, other_north, unit = ℂ^2, ℂ^2, ℂ^3, ℂ^2, ℂ^1
        bottom = randn(rng, ComplexF64, physical ⊗ physical' ← north ⊗ east ⊗ north' ⊗ east')
        top = randn(rng, ComplexF64, physical ⊗ physical' ← other_north ⊗ unit ⊗ other_north' ⊗ unit')
        reference = _layer_test_dense_product(convert(Array, bottom), convert(Array, top))
        exact = apply_PEPO_exact(bottom, top)
        @test ntuple(direction -> dim(domain(exact)[direction]), 4) == (4, 3, 4, 3)
        @test convert(Array, exact) ≈ reference
        @test convert(Array, combine_layers(bottom, top)) ≈ reference
    end

    @testset "Spatially varying unit cells" begin
        bottom = _layer_test_random_cell(rng, [1 2; 2 1], [2 1; 1 2])
        top = _layer_test_random_cell(rng, [2 1; 1 2], [1 2; 2 1])
        exact = apply_PEPO_exact(bottom, top)
        combined = combine_layers(bottom, top)
        @test size(exact) == size(combined) == (2, 2, 1)
        for row in 1:2, column in 1:2
            reference = _layer_test_dense_product(convert(Array, bottom[row, column]), convert(Array, top[row, column]))
            @test convert(Array, exact[row, column]) ≈ reference
            @test convert(Array, combined[row, column]) ≈ reference
        end
    end

    @testset "Compression without discarding bond states" begin
        physical, horizontal, unit = ℂ^2, ℂ^2, ℂ^1
        bottom = randn(rng, ComplexF64, physical ⊗ physical' ← unit ⊗ horizontal ⊗ unit' ⊗ horizontal')
        top = randn(rng, ComplexF64, physical ⊗ physical' ← unit ⊗ horizontal ⊗ unit' ⊗ horizontal')
        reference = _layer_test_periodic_row(top) * _layer_test_periodic_row(bottom)
        @test _layer_test_periodic_row(combine_layers(bottom, top)) ≈ reference

        compressed, info = combine_layers(bottom, top; truncation = notrunc(), return_info = true)
        @test _layer_test_periodic_row(compressed) ≈ reference rtol = 1.0e-9 atol = 1.0e-9
        @test isfinite(info.truncation_error)
        @test 0 <= info.truncation_error <= 1.0e-9
        @test size(info.horizontal) == size(info.vertical) == (1, 1)
        @test all(isfinite, info.horizontal) && all(isfinite, info.vertical)

        algorithm = PEPSKit.LocalTruncation(truncrank(4))
        compressed_cell, cell_info = combine_layers(
            InfinitePEPO(bottom), InfinitePEPO(top);
            truncation = algorithm, return_info = true
        )
        @test size(compressed_cell) == (1, 1, 1)
        @test _layer_test_periodic_row(compressed_cell[1, 1, 1]) ≈ reference rtol = 1.0e-9 atol = 1.0e-9
        @test isfinite(cell_info.truncation_error)
        bare = combine_layers(bottom, top; truncation = algorithm)
        @test bare isa PEPSKit.PEPOTensor
        @test _layer_test_periodic_row(bare) ≈ reference rtol = 1.0e-9 atol = 1.0e-9
    end

    @testset "Integer powers and independent copies" begin
        for method in (:sequential, :squaring)
            powered, info = combine_layers(A, 5; method, return_info = true)
            @test _layer_test_physical_matrix(powered) ≈ A_matrix^5
            @test info.method == method
            @test info.multiplications == (method == :sequential ? 4 : 3)
            @test length(info.steps) == info.multiplications
            @test all(step -> iszero(step.truncation_error), info.steps)

            cell = InfinitePEPO(A)
            single, single_info = combine_layers(cell, 1; method, return_info = true)
            @test iszero(single_info.multiplications)
            @test isempty(single_info.steps)
            @test single[1, 1, 1] !== cell[1, 1, 1]
            rmul!(single[1, 1, 1], 2)
            @test _layer_test_physical_matrix(cell[1, 1, 1]) ≈ A_matrix

            single_tensor = combine_layers(A, 1; method)
            rmul!(single_tensor, 2)
            @test _layer_test_physical_matrix(A) ≈ A_matrix

            powered_cell = combine_layers(InfinitePEPO(A), 5; method)
            @test _layer_test_physical_matrix(powered_cell[1, 1, 1]) ≈ A_matrix^5
        end
        compressed = combine_layers(A, 5; method = :squaring, truncation = truncrank(1))
        @test _layer_test_physical_matrix(compressed) ≈ A_matrix^5
    end

    @testset "Input compatibility" begin
        wrong_physical = _layer_test_product_operator(Matrix{Float64}(I, 3, 3))
        @test_throws SpaceMismatch combine_layers(A, wrong_physical)
        @test_throws SpaceMismatch combine_layers(InfinitePEPO(A), InfinitePEPO(wrong_physical))
        @test_throws DimensionMismatch combine_layers(InfinitePEPO(A; unitcell = (2, 1, 1)), InfinitePEPO(B))
        @test_throws ArgumentError combine_layers(InfinitePEPO(A; unitcell = (1, 1, 2)), InfinitePEPO(B))
        empty_cell = InfinitePEPO(Array{typeof(A)}(undef, 0, 1, 1))
        @test_throws ArgumentError combine_layers(empty_cell, empty_cell)
        @test_throws ArgumentError combine_layers(A, 0)
        @test_throws ArgumentError combine_layers(A, -1)
        @test_throws ArgumentError combine_layers(A, 2; method = :unknown)
    end
end
