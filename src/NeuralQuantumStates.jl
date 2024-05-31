module NeuralQuantumStates

using LinearAlgebra
using SparseArrays

const default_eltype = ComplexF64

include("hilbert.jl")
include("quantum_operator/quantum_operator.jl")
include("quantum_operator/helpers.jl")
include("quantum_operator/predefined_operators.jl")

end
