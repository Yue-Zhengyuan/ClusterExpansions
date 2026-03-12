module ClusterExpansions

using TensorKit, TensorKitTensors
using KrylovKit
using BlockTensorKit: ⊕, SumSpace
using MPSKit
using PEPSKit
using Graphs
using Zygote, OptimKit

include("utility/canonical_form.jl")
include("utility/observables.jl")

include("time_evolution/utility.jl")
include("time_evolution/local_truncation.jl")

include("models/models.jl")

include("time_evolution/time_evolve.jl")
include("models/time_evolve_models.jl")

include("clusterexpansions/generate_trees.jl")
include("clusterexpansions/generate_loops.jl")
include("clusterexpansions/clusters.jl")
include("clusterexpansions/contractions.jl")
include("clusterexpansions/PEPOs.jl")

include("clusterexpansions/solvers/linearsolvers.jl")
include("clusterexpansions/solvers/nonlinearsolvers_optimkit.jl")
include("clusterexpansions/solve_clusters.jl")

export EntanglementFiltering, filter_loop, truncate_loop

export Canonicalization, canonicalize
export PEPOObservable, PEPO_observables, localoperator_model, calculate_observables

export fidelity, apply_PEPO_exact
export NoEnvTruncation
export approximate_state

export ClusterExpansion, evolution_operator
export ising_operators
export spinless_fermion_operators, spinless_fermion_model
export heisenberg_operators, J1J2_operators
export tJ_operators, hubbard_operators

export evolution_operator
export StaticTimeEvolution, TimeDependentTimeEvolution
export UniformTimeEvolution, SquaringTimeEvolution
export UniformGroundStateTimeEvolution, SquaringGroundStateTimeEvolution
export time_evolve, get_time_array, time_scan
export time_evolve_model, time_scan_model

export clusterexpansion

end # module ClusterExpansions
