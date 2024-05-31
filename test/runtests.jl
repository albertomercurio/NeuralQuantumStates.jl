using NeuralQuantumStates
using Test
import Pkg

const GROUP = get(ENV, "GROUP", "All")

const testdir = dirname(@__FILE__)

const core_tests = ["quantum_operator.jl"]

if ((GROUP == "All") || (GROUP == "Code-Quality")) && (VERSION >= v"1.9")
    Pkg.add(["Aqua", "JET"])
    using Aqua
    using JET

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(NeuralQuantumStates; ambiguities = false)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(NeuralQuantumStates; target_defined_modules = true)
    end
end

if (GROUP == "All") || (GROUP == "Core")
    for test in core_tests
        include(joinpath(testdir, test))
    end
end
