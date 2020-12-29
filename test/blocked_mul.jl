@time @testset "Float64 Matrix-Vector" begin
    for n ∈ [10, 100, 500, 10000]
        for m ∈ [10, 100, 10000]
            u = zeros(n)
            A = rand(n, m)
            v = rand(m)
            @test blocked_mul!(u, A, v) ≈ A * v
            @test u ≈ blocked_mul(A, v)
        end
    end
end

@time @testset "ComplexF64 Matrix-Vector" begin
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

@time @testset "Float32 Matrix-Vector" begin
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

@time @testset "Int64 Matrix-Vector" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(Int, n)
            A = rand(-20:20, n, m)
            v = rand(-20:20, m)
            @test blocked_mul!(u, A, v) == A * v
            @test u == blocked_mul(A, v)
        end
    end
end

@time @testset "ComplexInt32 Matrix-Vector" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(Complex{Int32}, n) |> StructArray
            A = StructArray{Complex{Int32}}((rand(Int32.(-10:10), n, m), rand(Int32.(-10:10), n, m)))
            v = StructArray{Complex{Int32}}((rand(Int32.(-10:10),    m), rand(Int32.(-10:10),    m)))
            @test blocked_mul!(u, A, v) == collect(A) * collect(v)
            @test u == blocked_mul(A, v)
        end
    end
end

@time @testset "Float64 CoVector-Matrix" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(m)
            A = rand(n, m)
            v = rand(n)
            @test blocked_mul!(u', v', A) ≈ v' * A
            @test u' ≈ blocked_mul(v', A)

            @test blocked_mul!(transpose(u), transpose(v), A) ≈ transpose(v) * A
            @test transpose(u) ≈ blocked_mul(transpose(v), A)
        end
    end
end

@time @testset "ComplexF32 CoVector-Matrix" begin
    for n ∈ [10, 100, 500, 2000]
        for m ∈ [10, 100, 2000]
            u = zeros(Complex{Float32}, m)   |> StructArray
            A = rand(Complex{Float32}, n, m) |> StructArray
            v = rand(Complex{Float32}, n)    |> StructArray
            @test blocked_mul!(u', v', A) ≈ v' * A
            @test u' ≈ blocked_mul(v', A)

            @test blocked_mul!(transpose(u), transpose(v), A) ≈ transpose(v) * A
            @test transpose(u) ≈ blocked_mul(transpose(v), A)
        end
    end
end

@time @testset "ComplexFloat64 Matrix-Matrix" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m)))
            C2 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m)))
            A  = StructArray{ComplexF64}((randn(n, k), randn(n, k)))
            B  = StructArray{ComplexF64}((randn(k, m), randn(k, m)))

            @test blocked_mul!(C1, A, B) ≈ LinearAlgebra.mul!(C2, A, B)
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

            @test blocked_mul!(C1, A, B) ≈ LinearAlgebra.mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
        end
    end
end

@time @testset "Float64 Matrix-Matrix" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(n, m)
            C2 = zeros(n, m)
            A  = randn(n, k)
            B  = randn(k, m)
            At = copy(A')
            Bt = copy(B')

            @test blocked_mul!(C1, A, B) ≈ LinearAlgebra.mul!(C2, A, B)
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

            @test blocked_mul!(C1, A, B) ≈ LinearAlgebra.mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
            fill!(C1, NaN); @test blocked_mul!(C1, At', B)   ≈ C2
            fill!(C1, NaN); @test blocked_mul!(C1, A,   Bt') ≈ C2
            fill!(C1, NaN); @test blocked_mul!(C1, At', Bt') ≈ C2
        end
    end
end

@time @testset "Float32 Matrix-Matrix" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(Float32, n, m)
            C2 = zeros(Float32, n, m)
            A  = randn(Float32, n, k)
            B  = randn(Float32, k, m)
            At = copy(A')
            Bt = copy(B')

            @test blocked_mul!(C1, A, B) ≈ LinearAlgebra.mul!(C2, A, B)
            @test blocked_mul(A, B) ≈ C1
            fill!(C1, NaN32); @test blocked_mul!(C1, At', B)   ≈ C2
            fill!(C1, NaN32); @test blocked_mul!(C1, A,   Bt') ≈ C2
            fill!(C1, NaN32); @test blocked_mul!(C1, At', Bt') ≈ C2
        end
    end
end

@time @testset "Int64 Matrix-Matrix" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(Int64, n, m)
            C2 = zeros(Int64, n, m)
            A  = rand(Int64.(-100:100), n, k)
            B  = rand(Int64.(-100:100), k, m)
            At = copy(A')
            Bt = copy(B')

            @test blocked_mul!(C1, A, B) == LinearAlgebra.mul!(C2, A, B)
            @test blocked_mul(A, B) == C1
            fill!(C1, typemax(Int64)); @test blocked_mul!(C1, At', B)   == C2
            fill!(C1, typemax(Int64)); @test blocked_mul!(C1, A,   Bt') == C2
            fill!(C1, typemax(Int64)); @test blocked_mul!(C1, At', Bt') == C2
        end
    end
end

@time @testset "Int32 Matrix-Matrix" begin
    for sz ∈ [10, 50, 100, 200, 400, 1000]
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m)" begin
            C1 = zeros(Int32, n, m)
            C2 = zeros(Int32, n, m)
            A  = rand(Int32.(-100:100), n, k)
            B  = rand(Int32.(-100:100), k, m)
            At = copy(A')
            Bt = copy(B')

            @test blocked_mul!(C1, A, B) == LinearAlgebra.mul!(C2, A, B)
            @test blocked_mul(A, B) == C1
            fill!(C1, typemax(Int32)); @test blocked_mul!(C1, At', B)   == C2
            fill!(C1, typemax(Int32)); @test blocked_mul!(C1, A,   Bt') == C2
            fill!(C1, typemax(Int32)); @test blocked_mul!(C1, At', Bt') == C2
        end
    end
end
