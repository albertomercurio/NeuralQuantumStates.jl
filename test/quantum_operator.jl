@testset "Quantum Operator" begin
    # The following is equivalent to hi = Hilbert((2, 2, 2))
    # But it is needed just for runtests
    hi = Hilbert((2,))
    hi = hi * (hi^4)

    mat_x = [0 1; 1 0]
    mat_y = [0 1im; -1im 0]
    mat_z = [-1 0; 0 1]

    # op1 = QuantumOperator(hi, 1, mat_x) * QuantumOperator(hi, 2, mat_x)
    # op2 = QuantumOperator(hi, 2, mat_z) * QuantumOperator(hi, 3, mat_z)

    for type in [Int32, Int64, Float32, Float64, ComplexF32, ComplexF64]
        op1 = sigmax(hi, 1, type) * sigmax(hi, 2, type)
        op2 = sigmaz(hi, 2, type) * sigmaz(hi, 3, type)

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
        @test res.dict[(1, 2)] == -op1.dict[(1, 2)]
        @test res.constant[] == 3.0

        res = -op1
        @test res.dict[(1, 2)] == -op1.dict[(1, 2)]

        res = op1 - op1
        @test length(res.dict) == 1
        @test iszero(sum(res.dict[1, 2]))
        @test iszero(res.constant[])

        res = op1 + op2

        @test length(res.dict) == 2
        @test res.dict[(1, 2)] == kron(mat_x, mat_x)
        @test res.dict[(2, 3)] == kron(mat_z, mat_z)

        res = 2.0 * (op1 + 1.2)
        @test res.dict[(1, 2)] == 2.0 * op1.dict[(1, 2)]
        @test res.constant[] == 2.0 * 1.2

        res = (op1 + 1.2) * 2.0
        @test res.dict[(1, 2)] == 2.0 * op1.dict[(1, 2)]
        @test res.constant[] == 2.0 * 1.2

        res = 2.0 * op1 / 2
        @test res.dict[(1, 2)] == op1.dict[(1, 2)]

        hi1 = Hilbert((2, 2, 2))
        hi2 = Hilbert((2, 2, 4))

        @test_throws ArgumentError QuantumOperator(hi1, 1, mat_x) + QuantumOperator(hi2, 1, mat_x)

        res = op1 * op2
        @test length(res.dict) == 1
        @test res.dict[(1, 2, 3)] == kron(mat_x, mat_x * mat_z, mat_z)

        res = (op1 + op2 + 1.2) * op1
        @test length(res.dict) == 2
        @test res.dict[(1, 2)] == op1.dict[(1, 2)]^2 + 1.2 * op1.dict[(1, 2)]
        @test res.dict[(1, 2, 3)] == kron(mat_x, mat_z * mat_x, mat_z)

        op3 = sigmaz(hi, 4, type) * sigmax(hi, 5, type)
        res = op1 * op2 * op3 + op3
        @test length(res.dict) == 2
        @test res.dict[(1, 2, 3, 4, 5)] == kron(mat_x, mat_x * mat_z, mat_z, mat_z, mat_x)
        @test res.dict[(4, 5)] == kron(mat_z, mat_x)

        if type <: Complex
            op = sigmay(hi, 1, type) * sigmax(hi, 2, type)
            res = op * op
            @test length(res.dict) == 1
            @test res.dict[(1, 2)] == kron(mat_y * mat_y, mat_x * mat_x)

            res = op1 * op2 * op
            @test length(res.dict) == 1
            @test res.dict[(1, 2, 3)] == kron(mat_x * mat_y, mat_x * mat_z * mat_x, mat_z)
        end
    end
end