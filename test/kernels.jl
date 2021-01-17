@time @testset "kernels" begin
    @testset begin
        A = [1 2; 3 4]
        B = [5 6; 7 8]
        C1 = zeros(Int, 2, 2)
        C2 = zeros(Int, 2, 2)
        Gaius.add_gemm_kernel!(C1, A, B, Val(1))
        LinearAlgebra.mul!(C2, A, B)
        @test C1 == C2
    end

    @testset begin
        A = [1 2; 3 4]
        v = [5; 6]
        u1 = zeros(Int, 2)
        u2 = zeros(Int, 2)
        Gaius.add_gemm_kernel!(u1, A, v)
        LinearAlgebra.mul!(u2, A, v)
        @test u1 == u2
    end

    @testset begin
        v = adjoint([5; 6])
        A = [1 2; 3 4]
        u1 = adjoint(zeros(Int, 2))
        u2 = adjoint(zeros(Int, 2))
        Gaius.add_gemm_kernel!(u1, v, A)
        LinearAlgebra.mul!(u2, v, A)
        @test u1 == u2
    end

    @testset begin
        A = [1 2; 3 4]
        B = [5 6; 7 8]
        C = zeros(Int, 2, 2)
        Gaius.add_gemm_kernel!(C, A, B, Val(1))
        @test C == A * B
    end
end
