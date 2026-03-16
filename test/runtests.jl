using SafeTestsets

@time @safetestset "Phase transition - Ising model" begin
    include("phase_transition_ising.jl")
end
