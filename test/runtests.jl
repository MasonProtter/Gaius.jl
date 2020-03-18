using Test, LinearAlgebra, Random
using Gaius
using StructArrays

# @testset "Matrix-Vector products" begin
    

# end


@testset "ComplexFloat64 Matrix Multiplication" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m))) 
            C2 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m))) 
            A  = StructArray{ComplexF64}((randn(n, k), randn(n, k))) 
            B  = StructArray{ComplexF64}((randn(k, m), randn(k, m)))
           
            @test blocked_mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
        end
    end
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = shuffle([sz + rand(-5:5), sz + rand(-5:5), 10])
        @testset "($n × $k) × ($k × $m)" begin
            C1 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m))) 
            C2 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m))) 
            A  = StructArray{ComplexF64}((randn(n, k), randn(n, k))) 
            B  = StructArray{ComplexF64}((randn(k, m), randn(k, m)))
           
            @test blocked_mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
        end
    end
end

@testset "Float64 Multiplication" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(n, m)
            C2 = zeros(n, m)
            A  = randn(n, k)
            B  = randn(k, m)
            At = copy(A')
            Bt = copy(B')
            
            @test blocked_mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
            
            fill!(C1, NaN); @test blocked_mul!(C1, At', B)   ≈ C2
            fill!(C1, NaN); @test blocked_mul!(C1, A,   Bt') ≈ C2
            fill!(C1, NaN); @test blocked_mul!(C1, At', Bt') ≈ C2
        end
    end
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = shuffle([sz + rand(-5:5), sz + rand(-5:5), 10])
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(n, m)
            C2 = zeros(n, m)
            A  = randn(n, k)
            B  = randn(k, m)
            At = copy(A')
            Bt = copy(B')
           
            @test blocked_mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
            fill!(C1, NaN); @test blocked_mul!(C1, At', B)   ≈ C2
            fill!(C1, NaN); @test blocked_mul!(C1, A,   Bt') ≈ C2
            fill!(C1, NaN); @test blocked_mul!(C1, At', Bt') ≈ C2
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
            At = copy(A')
            Bt = copy(B')
           
            @test blocked_mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
            fill!(C1, NaN32); @test blocked_mul!(C1, At', B)   ≈ C2
            fill!(C1, NaN32); @test blocked_mul!(C1, A,   Bt') ≈ C2
            fill!(C1, NaN32); @test blocked_mul!(C1, At', Bt') ≈ C2
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
            At = copy(A')
            Bt = copy(B')
           
            @test blocked_mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
            fill!(C1, typemax(Int64)); @test blocked_mul!(C1, At', B)   ≈ C2
            fill!(C1, typemax(Int64)); @test blocked_mul!(C1, A,   Bt') ≈ C2
            fill!(C1, typemax(Int64)); @test blocked_mul!(C1, At', Bt') ≈ C2
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
            At = copy(A')
            Bt = copy(B')

            @test blocked_mul!(C1, A, B) ≈ mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
            fill!(C1, typemax(Int32)); @test blocked_mul!(C1, At', B)   ≈ C2
            fill!(C1, typemax(Int32)); @test blocked_mul!(C1, A,   Bt') ≈ C2
            fill!(C1, typemax(Int32)); @test blocked_mul!(C1, At', Bt') ≈ C2
        end
    end
end
