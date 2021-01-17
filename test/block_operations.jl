@time @testset "block_operations" begin
    @testset begin
        A = [1 2; 3 4]
        B = [5 6; 7 8]
        C = Matrix{Int}(undef, 2, 2)
        Gaius.block_covec_vec_mul!(Gaius.multithreaded, C, A, B, 1)
        @test C == A * B
    end

    @testset begin
        A = [1 2; 3 4]
        B = [5, 6]
        C = Vector{Int}(undef, 2)
        Gaius.block_covec_vec_mul!(Gaius.multithreaded, C, A, B, 1)
        @test C == A * B
    end
end
