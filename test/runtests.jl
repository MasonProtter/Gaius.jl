using Test, LinearAlgebra
using Gaius

@testset "Float64 Multiplication" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(n, m)
            C2 = zeros(n, m)
            A  = randn(n, k)
            B  = randn(k, m)
           
            @test Gaius.mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test Gaius.:(*)(A, B) ≈ C1
        end
    end
end

@testset "Float32 Multiplication" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(Float32, n, m)
            C2 = zeros(Float32, n, m)
            A  = randn(Float32, n, k)
            B  = randn(Float32, k, m)
           
            @test Gaius.mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test Gaius.:(*)(A, B) ≈ C1
        end
    end
end


@testset "Int64 Multiplication" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(Int, n, m)
            C2 = zeros(Int, n, m)
            A  = rand(-100:100, n, k)
            B  = rand(-100:100, k, m)
           
            @test Gaius.mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test Gaius.:(*)(A, B) ≈ C1
        end
    end
end


@testset "Int32 Multiplication" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(Int32, n, m)
            C2 = zeros(Int32, n, m)
            A  = rand(Int32.(-100:100), n, k)
            B  = rand(Int32.(-100:100), k, m)
           
            @test Gaius.mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test Gaius.:(*)(A, B) ≈ C1
        end
    end
end
