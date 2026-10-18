using LinearAlgebra

# One Julia process runs all focused testsets with one BLAS thread.
BLAS.set_num_threads(1)

module ConstructionTests
    include("helpers.jl")
    include("construction.jl")
end

module LayerTests
    include("layers.jl")
end

module SolverTests
    include("helpers.jl")
    include("solvers.jl")
end

module IntegrationTests
    include("helpers.jl")
    include("integration.jl")
end

module FermionTests
    include("helpers.jl")
    include("fermions.jl")
end
