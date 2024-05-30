@testset "Quantum Operator" begin
    hi = Hilbert((2, 2, 2, 2, 2))
    ao = 1
    mat_x = [0 1; 1 0]
    mat_z = [1 0; 0 -1]

    op1 = QuantumOperator(hi, 1, mat_x) * QuantumOperator(hi, 2, mat_x)
    op2 = QuantumOperator(hi, 2, mat_z) * QuantumOperator(hi, 3, mat_z)

    res = op1 + op2

    @test length(res.dict) == 2
    @test res.dict[(1,2)] == kron(mat_x, mat_x)
    @test res.dict[(2,3)] == kron(mat_z, mat_z)
end