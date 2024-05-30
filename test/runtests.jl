using NeuralQuantumStates
using Test
import Pkg

const GROUP = get(ENV, "GROUP", "All")

if ((GROUP == "All") || (GROUP == "Code-Quality")) && (VERSION >= v"1.9")
    Pkg.add(["Aqua", "JET"])
    using Aqua
    using JET
    @testset "NeuralQuantumStates.jl" begin
        @testset "Code quality (Aqua.jl)" begin
            Aqua.test_all(NeuralQuantumStates; ambiguities = false)
        end
        @testset "Code linting (JET.jl)" begin
            JET.test_package(NeuralQuantumStates; target_defined_modules = true)
        end
        # Write your tests here.
    end
end
