module ClusterExpansions

using TensorKit
using LinearAlgebra
using MatrixAlgebraKit: TruncationStrategy
using KrylovKit: LSMR, lssolve
using BlockTensorKit: SumSpace, spzeros
using PEPSKit: InfinitePEPS, InfinitePEPO
import PEPSKit
using Graphs
using Random

import TensorKitTensors.SpinOperators as SO
import TensorKitTensors.FermionOperators as FO

include("utility.jl")
include("clusterexpansions/spaces.jl")
include("clusterexpansions/generate_trees.jl")
include("clusterexpansions/generate_loops.jl")
include("clusterexpansions/clusters.jl")
include("clusterexpansions/generate_clusters.jl")
include("clusterexpansions/ce_alg.jl")
include("clusterexpansions/PEPOs.jl")
include("clusterexpansions/contractions.jl")

include("clusterexpansions/solvers/linearsolvers.jl")
include("clusterexpansions/solvers/loopsolvers.jl")
include("clusterexpansions/solve_clusters.jl")

include("time_evolution/layers.jl")
include("models/models.jl")

export ClusterExpansion
export clusterexpansion, evolution_operator
export apply_PEPO_exact, combine_layers
export ising_operators, heisenberg_operators, spinless_fermion_operators

end # module ClusterExpansions
