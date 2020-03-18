using Test, LinearAlgebra, Random
using Gaius
using StructArrays

@testset "Matrix-Vector Float64" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(n)
            A = rand(n, m)
            v = rand(m)
            @test blocked_mul!(u, A, v) ≈ A * v
            @test u ≈ blocked_mul(A, v)
        end
    end
end

@testset "Matrix-Vector ComplexF64" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(ComplexF64, n)   |> StructArray
            A = rand(ComplexF64, n, m) |> StructArray
            v = rand(ComplexF64, m)    |> StructArray
            @test blocked_mul!(u, A, v) ≈ collect(A) * collect(v)
            @test u ≈ blocked_mul(A, v)
        end
    end
end

@testset "Matrix-Vector Float32" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(Float32, n)
            A = rand(Float32, n, m)
            v = rand(Float32, m)
            @test blocked_mul!(u, A, v) ≈ A * v
            @test u ≈ blocked_mul(A, v)
        end
    end
end

@testset "Matrix-Vector Int64" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(Int, n)
            A = rand(-20:20, n, m)
            v = rand(-20:20, m)
            @test blocked_mul!(u, A, v) ≈ A * v
            @test u ≈ blocked_mul(A, v)
        end
    end
end

@testset "Matrix-Vector ComplexInt32" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(Complex{Int32}, n) |> StructArray       
            A = StructArray{Complex{Int32}}((rand(Int32.(-10:10), n, m), rand(Int32.(-10:10), n, m)))
            v = StructArray{Complex{Int32}}((rand(Int32.(-10:10),    m), rand(Int32.(-10:10),    m)))   
            @test blocked_mul!(u, A, v) ≈ collect(A) * collect(v)
            @test u ≈ blocked_mul(A, v)
        end
    end
end



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
