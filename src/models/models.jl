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
    end
    return spaces
end

function ising_operators(J, g, z; spin_symmetry = Trivial, T = ComplexF64, loop_space = nothing, kwargs...)
    twosite_op = rmul!(SO.SᶻSᶻ(T, spin_symmetry), -4 * J)
    if spin_symmetry == Trivial
        onesite_op = rmul!(SO.σˣ(T), g * -J) + rmul!(SO.σᶻ(T), z * -J)
    elseif spin_symmetry == Z2Irrep
        @assert (g == 0) && (z == 0) "Z2-symmetric Ising model can not have a magnetic field. g = $g and z = $z"
        pspace = Z2Space(0 => 1, 1 => 1)
        onesite_op = rmul!(id(pspace), 0.0)
    end
    spaces = spaces_ising(spin_symmetry, g == 0; loop_space)
    return ClusterExpansion(twosite_op, onesite_op; spaces, kwargs...)
end
