struct ClusterExpansion
    twosite_op
    onesite_op
    p
    verbosity
    T
    spaces
    solving_loops
    svd
end

function ClusterExpansion(twosite_op, onesite_op; p = 3, verbosity = 0, T = ComplexF64, spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10, solving_loops = true, svd = true)
    return ClusterExpansion(twosite_op, onesite_op, p, verbosity, T, spaces, solving_loops, svd)
end
