"""Choose reduced classical or full quantum spaces for the spin-1/2 Ising model."""
function spaces_ising(spin_symmetry::Type{<:Sector}, smaller_spaces::Bool; loop_space = nothing)
    spin_symmetry in (Trivial, Z2Irrep) || throw(ArgumentError("Ising supports Trivial or Z2Irrep spin symmetry."))
    pspace = SO.spin_space(spin_symmetry)
    default_loop = spin_symmetry == Trivial ? ℂ^10 : Z2Space(0 => 5, 1 => 5)
    loops = _select_loop_space(pspace, loop_space, default_loop)
    return smaller_spaces ? _LevelSpaces(pspace, loops) : _operator_spaces(pspace; loop_space = loops)
end

"""
    ising_operators(J, g, z; spin_symmetry=Trivial, T=ComplexF64, kwargs...)

Configure the spin-1/2 Ising Hamiltonian with bond term ``-J σ^z_i σ^z_j`` and site term ``-J (g σ^x_i + z σ^z_i)``.
`Z2Irrep` retains the spin-flip symmetry and permits the transverse field `g`, while requiring `z == 0`.
Classical models (`g == 0`) use virtual dimensions ``2^level``; quantum models use full operator-Schmidt spaces.
Pass `spaces` or `loop_space` to choose different virtual spaces, and forward other keywords to `ClusterExpansion`.
"""
function ising_operators(
        J::Number, g::Number, z::Number;
        spin_symmetry::Type{<:Sector} = Trivial, T::Type{<:Number} = ComplexF64,
        loop_space = nothing, spaces = nothing, kwargs...,
    )
    spin_symmetry in (Trivial, Z2Irrep) || throw(ArgumentError("Ising supports Trivial or Z2Irrep spin symmetry."))
    spin_symmetry == Z2Irrep && !iszero(z) && throw(ArgumentError("A longitudinal field breaks the Ising Z2 spin-flip symmetry; set z = 0."))
    twosite_op = (-4 * T(J)) * SO.SᶻSᶻ(T, spin_symmetry)
    onesite_op = (-T(J) * T(g)) * SO.σˣ(T, spin_symmetry)
    if spin_symmetry == Trivial && !iszero(z)
        onesite_op += (-T(J) * T(z)) * SO.σᶻ(T, Trivial)
    end
    selected_spaces = isnothing(spaces) ? spaces_ising(spin_symmetry, iszero(g); loop_space) : spaces
    return ClusterExpansion(twosite_op, onesite_op; T, spaces = selected_spaces, kwargs...)
end

function ising_operators(; J::Number = 1.0, g::Number = 0.0, z::Number = 0.0, kwargs...)
    return ising_operators(J, g, z; kwargs...)
end

"""
    spinless_fermion_operators(t, V, μ; b=0, δ=0, T=ComplexF64, kwargs...)

Configure spinless fermions with hopping ``-t (f^+_i f^-_j - f^-_i f^+_j)``, density interaction ``V (n_i - 1/2) (n_j - 1/2)``, and site term ``(2 V b - μ) n_i``.
The optional pairing term is ``-δ (f^-_i f^-_j - f^+_i f^+_j)``.
Fermion-parity grading is retained, including when `δ` is nonzero.
Virtual spaces contain the full operator-Schmidt sectors unless `spaces` is supplied; `loop_space` selects the space reserved for loops.
Other keywords are forwarded to `ClusterExpansion`.
"""
function spinless_fermion_operators(
        t::Number, V::Number, μ::Number;
        b::Number = 0.0, δ::Number = 0.0, T::Type{<:Number} = ComplexF64,
        loop_space = nothing, spaces = nothing, kwargs...,
    )
    number_operator = FO.f_num(T)
    pspace = domain(number_operator)[1]
    shifted_number = number_operator - id(T, pspace) / 2
    twosite_op = (-T(t)) * FO.f_hop(T) + T(V) * (shifted_number ⊗ shifted_number)
    if !iszero(δ)
        twosite_op -= T(δ) * (FO.f⁻f⁻(T) - FO.f⁺f⁺(T))
    end
    onesite_op = (2 * T(V) * T(b) - T(μ)) * number_operator
    selected_spaces = isnothing(spaces) ? _operator_spaces(pspace; loop_space) : spaces
    return ClusterExpansion(twosite_op, onesite_op; T, spaces = selected_spaces, kwargs...)
end

function spinless_fermion_operators(; t::Number = 1.0, V::Number = 0.0, μ::Number = 0.0, kwargs...)
    return spinless_fermion_operators(t, V, μ; kwargs...)
end

"""Validate the spin and parameter restrictions imposed by the chosen spin symmetry."""
function _check_heisenberg_parameters(
        Jx::Number, Jy::Number, Jz::Number, h::Number,
        spin::Real, spin_symmetry::Type{<:Sector},
    )
    isfinite(spin) && spin >= 0 && isinteger(2 * spin) || throw(ArgumentError("spin must be a nonnegative integer or half-integer."))
    if spin_symmetry == Z2Irrep
        spin == 1 // 2 || throw(ArgumentError("Z2Irrep is supported only for spin 1/2."))
        iszero(h) || throw(ArgumentError("A longitudinal field breaks the Z2 spin-flip symmetry; set h = 0."))
    elseif spin_symmetry == U1Irrep
        Jx == Jy || throw(ArgumentError("U1Irrep requires Jx == Jy."))
    elseif spin_symmetry == SU2Irrep
        Jx == Jy == Jz && iszero(h) || throw(ArgumentError("SU2Irrep requires Jx == Jy == Jz and h == 0."))
    elseif spin_symmetry != Trivial
        throw(ArgumentError("Heisenberg supports Trivial, Z2Irrep, U1Irrep, or SU2Irrep spin symmetry."))
    end
    return nothing
end

"""
    heisenberg_operators(Jx, Jy, Jz, h; spin=1//2, spin_symmetry=Trivial, T=ComplexF64, kwargs...)

Configure the XYZ spin Hamiltonian with bond term ``J_x S^x_i S^x_j + J_y S^y_i S^y_j + J_z S^z_i S^z_j`` and site term ``h S^z_i``.
`U1Irrep` requires `Jx == Jy` and permits the longitudinal field.
`SU2Irrep` requires isotropic exchange and zero field; `Z2Irrep` permits XYZ exchange for spin 1/2 with zero field.
Virtual spaces contain the full operator-Schmidt sectors unless `spaces` is supplied; `loop_space` selects the space reserved for loops.
Other keywords are forwarded to `ClusterExpansion`.
"""
function heisenberg_operators(
        Jx::Number, Jy::Number, Jz::Number, h::Number;
        spin::Real = 1 // 2, spin_symmetry::Type{<:Sector} = Trivial,
        T::Type{<:Number} = ComplexF64, loop_space = nothing, spaces = nothing, kwargs...,
    )
    _check_heisenberg_parameters(Jx, Jy, Jz, h, spin, spin_symmetry)
    if spin_symmetry == SU2Irrep
        twosite_op = T(Jx) * SO.S_exchange(T, SU2Irrep; spin)
    elseif spin_symmetry == U1Irrep
        twosite_op = (T(Jx) / 2) * (SO.S⁺S⁻(T, U1Irrep; spin) + SO.S⁻S⁺(T, U1Irrep; spin)) +
            T(Jz) * SO.SᶻSᶻ(T, U1Irrep; spin)
    else
        twosite_op = T(Jx) * SO.SˣSˣ(T, spin_symmetry; spin) +
            T(Jy) * SO.SʸSʸ(T, spin_symmetry; spin) +
            T(Jz) * SO.SᶻSᶻ(T, spin_symmetry; spin)
    end
    pspace = domain(twosite_op)[1]
    onesite_op = iszero(h) ? zeros(T, pspace ← pspace) : T(h) * SO.Sᶻ(T, spin_symmetry; spin)
    selected_spaces = isnothing(spaces) ? _operator_spaces(pspace; loop_space) : spaces
    return ClusterExpansion(twosite_op, onesite_op; T, spaces = selected_spaces, kwargs...)
end

function heisenberg_operators(;
        Jx::Number = 1.0, Jy::Number = Jx, Jz::Number = Jx, h::Number = 0.0, kwargs...,
    )
    return heisenberg_operators(Jx, Jy, Jz, h; kwargs...)
end
