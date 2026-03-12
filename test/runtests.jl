using SafeTestsets

@time @safetestset "Phase transition - Ising model" begin
    include("phase_transition_ising.jl")
end

@time @safetestset "Phase transition - Spinless Fermion model" begin
    include("phase_transition_spinless_fermions.jl")
end
