module NeuralQuantumStates

using LinearAlgebra
using SparseArrays

using SparseArrays: AbstractSparseMatrixCSC, getcolptr

const default_eltype = ComplexF64

include("hilbert.jl")
include("quantum_operator/quantum_operator.jl")
include("quantum_operator/helpers.jl")
include("quantum_operator/predefined_operators.jl")
include("quantum_operator/connected_states.jl")

end
