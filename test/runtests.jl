using Test, LoopVectorizationBLAS

@testset "multiplication" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(n, m)
            C2 = zeros(n, m)
            A  = randn(n, k)
            B  = randn(k, m)
            
            blocked_mul!(C1, A, B)
            mul!(        C2, A, B)

            @test C1 ≈ C2
        end
    end
end

# @testset "multiplication" begin
#     for sz ∈ [10, 50, 100, 200, 400]
#         n, k, m = (sz .+ rand(-5:5, 3))
#         C1 = zeros(n, m)
#         C2 = zeros(n, m)
#         A  = randn(n, k)
#         B  = randn(k, m)
        
#         blocked_mul!(C1, A, B)
#         mul!(        C2, A, B)

#         @test lvbmul(A, B) == A*B
#     end
# end
