struct ClusterExpansion
    twosite_op
    onesite_op
    nn_term
    p
    verbosity
    T
    spaces
    solving_loops
    svd
    envspace
end

_envspace(::ComplexSpace) = χ -> ℂ^χ
_envspace(::GradedSpace{ZNIrrep{N}, NTuple{N, Int}}) where {N} = χ -> ZNSpace{N}(0 => χ - (N - 1) * div(χ, N), [i => div(χ, N) for i in 1:(N - 1)]...)
_envspace(::GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}) = χ -> Vect[U1Irrep](0 => χ - 2 * div(χ, 4), 1 => div(χ, 4), -1 => div(χ, 4))
_envspace(::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}) = χ -> Vect[SU2Irrep](0 => χ - 2 * div(χ, 4), 1 // 2 => div(χ, 4), 1 => div(χ, 4))
_envspace(::GradedSpace{FermionParity, Tuple{Int64, Int64}}) = χ -> Vect[FermionParity](0 => χ - div(χ, 2), 1 => div(χ, 2))
_envspace(space::GradedSpace{ProductSector{T}, T2}) where {T <: Tuple, T2} = χ -> Vect[sectortype(space)](ntuple(_ -> 0, fieldcount(T)) => χ)

function ClusterExpansion(twosite_op, onesite_op; nn_term = nothing, p = 3, verbosity = 0, T = ComplexF64, spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10, solving_loops = true, svd = true, envspace = χ -> ℂ^χ)
    return ClusterExpansion(twosite_op, onesite_op, nn_term, p, verbosity, T, spaces, solving_loops, svd, envspace)
end

function spaces_ising(spin_symmetry, smaller_spaces; loop_space = nothing)
    if spin_symmetry == Trivial
        if isnothing(loop_space)
            loop_space = ℂ^10
        end
        if smaller_spaces
            spaces = i -> (i >= 0) ? ℂ^(2^(i)) : loop_space
        else
            spaces = i -> (i >= 0) ? ℂ^(2^(2 * i)) : loop_space
        end
        envspace = χ -> ℂ^χ
    elseif spin_symmetry == Z2Irrep
        if isnothing(loop_space)
            loop_space = Z2Space(0 => 5, 1 => 5)
        end
        spaces = i -> if i == 0
            Z2Space(0 => 1)
        elseif i > 0
            Z2Space(0 => 2^(i - 1), 1 => 2^(i - 1))
        else
            loop_space
        end
        envspace = χ -> Z2Space(0 => χ - div(χ, 2), 1 => div(χ, 2))
    end
    return spaces, envspace
end

function ising_operators(J, g, z; spin_symmetry = Trivial, T = ComplexF64, loop_space = nothing, kwargs...)
    twosite_op = rmul!(PEPSKit.σᶻᶻ(T, spin_symmetry), -J)
    if spin_symmetry == Trivial
        onesite_op = rmul!(PEPSKit.σˣ(T), g * -J) + rmul!(PEPSKit.σᶻ(T), z * -J)
    elseif spin_symmetry == Z2Irrep
        @assert (g == 0) && (z == 0) "Z2-symmetric Ising model can not have a magnetic field. g = $g and z = $z"
        pspace = Z2Space(0 => 1, 1 => 1)
        onesite_op = rmul!(id(pspace), 0.0)
    end
    spaces, envspace = spaces_ising(spin_symmetry, g == 0; loop_space)
    return ClusterExpansion(twosite_op, onesite_op; spaces, envspace, kwargs...)
end

function spinless_fermion_operators(t, V, μ; b = 0.0, δ = 0.0, T = ComplexF64, loop_space = Vect[fℤ₂](0 => 5, 1 => 5), kwargs...)
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)

    kinetic_operator = FermionOperators.f_hop(T)
    number_operator = FermionOperators.f_num(T)
    number_operator_halffilling = number_operator - id(pspace) / 2
    symmetry_breaking_term = FermionOperators.f⁻f⁻(T) - FermionOperators.f⁺f⁺(T)
    @tensor number_twosite[-1 -2; -3 -4] := number_operator_halffilling[-1; -3] * number_operator_halffilling[-2; -4]

    twosite_op = rmul!(kinetic_operator, -T(t)) + rmul!(number_twosite, T(V)) - rmul!(symmetry_breaking_term, T(δ))
    onesite_op = rmul!(number_operator, -T(μ)) + rmul!(number_operator, 2 * T(V) * T(b))

    if t == 0.0
        spaces = i -> if i == 0
            Vect[fℤ₂](0 => 1)
        elseif i > 0
            Vect[fℤ₂](0 => 2^(i - 1), 1 => 2^(i - 1))
        else
            loop_space
        end
    else
        spaces = i -> if i == 0
            Vect[fℤ₂](0 => 1)
        elseif i > 0
            Vect[fℤ₂](0 => 2^(2 * i - 1), 1 => 2^(2 * i - 1))
        else
            loop_space
        end
    end
    envspace = χ -> Vect[fℤ₂](0 => div(χ, 2), 1 => div(χ, 2))
    return ClusterExpansion(twosite_op, onesite_op; T, spaces, envspace, kwargs...)
end
