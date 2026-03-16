module ClusterExpansions

using TensorKit
using MatrixAlgebraKit: TruncationStrategy
using KrylovKit
using BlockTensorKit: ⊞, SumSpace
using MPSKit
using PEPSKit
using Graphs

import TensorKitTensors.SpinOperators as SO
import TensorKitTensors.FermionOperators as FO

include("clusterexpansions/ce_alg.jl")
include("clusterexpansions/generate_trees.jl")
include("clusterexpansions/clusters.jl")
include("clusterexpansions/contractions.jl")
include("clusterexpansions/PEPOs.jl")

include("clusterexpansions/solvers/linearsolvers.jl")
include("clusterexpansions/solve_clusters.jl")

include("models/models.jl")

export EntanglementFiltering, filter_loop, truncate_loop

export PEPOObservable, PEPO_observables, localoperator_model, calculate_observables

export fidelity, apply_PEPO_exact

export ClusterExpansion
export clusterexpansion, evolution_operator

export ising_operators

end # module ClusterExpansions
