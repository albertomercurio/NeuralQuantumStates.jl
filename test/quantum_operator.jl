@testset "Quantum Operator" begin
    hi = Hilbert((2, 2, 2, 2, 2))
    mat_x = [0 1; 1 0]
    mat_z = [1 0; 0 -1]

    op1 = QuantumOperator(hi, 1, mat_x) * QuantumOperator(hi, 2, mat_x)
    op2 = QuantumOperator(hi, 2, mat_z) * QuantumOperator(hi, 3, mat_z)

    res = op1 + 2.0
    @test res.dict == op1.dict
    @test res.constant[] == 2.0

    res = 3.0 + op1
    @test res.dict == op1.dict
    @test res.constant[] == 3.0

    res = op1 - 2.0
    @test res.dict == op1.dict
    @test res.constant[] == -2.0

    res = 3.0 - op1
    @test res.dict[(1, 2)] == - op1.dict[(1, 2)]
    @test res.constant[] == 3.0

    res = -op1
    @test res.dict[(1, 2)] == - op1.dict[(1, 2)]

    res = op1 - op1
    @test length(res.dict) == 1
    @test iszero(sum(res.dict[1, 2]))
    @test iszero(res.constant[])

    res = op1 + op2

    @test length(res.dict) == 2
    @test res.dict[(1,2)] == kron(mat_x, mat_x)
    @test res.dict[(2,3)] == kron(mat_z, mat_z)

    res = 2.0 * (op1 + 1.2)
    @test res.dict[(1, 2)] == 2.0 * op1.dict[(1, 2)]
    @test res.constant[] == 2.0 * 1.2

    res = (op1 + 1.2) * 2.0
    @test res.dict[(1, 2)] == 2.0 * op1.dict[(1, 2)]
    @test res.constant[] == 2.0 * 1.2

    hi1 = Hilbert((2, 2, 2))
    hi2 = Hilbert((2, 2, 4))

    @test_throws ArgumentError QuantumOperator(hi1, 1, mat_x) + QuantumOperator(hi2, 1, mat_x)

    res = op1 * op2
    @test length(res.dict) == 1
    @test res.dict[(1, 2, 3)] == kron(mat_x, mat_x * mat_z, mat_z)

    res = (op1 + op2 + 1.2) * op1
    @test res.dict[(1, 2)] == op1.dict[(1, 2)]^2 + 1.2 * op1.dict[(1, 2)]
    @test res.dict[(1, 2, 3)] == kron(mat_x, mat_z * mat_x, mat_z)
end