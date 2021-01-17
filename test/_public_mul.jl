# The following variables need to be defined before `include`-ing this file:
# `testset_name_suffix`
# `n_values_1`
# `n_values_2`
# `m_values_1`
# `m_values_2`
# `sz_values`

@time @testset "Float64 Matrix-Vector $(testset_name_suffix)" begin
    for n ∈ n_values_1
        for m ∈ m_values_1
            u1 = zeros(n)
            u2 = zeros(n)
            A = rand(n, m)
            v = rand(m)
            @test Gaius.mul!(u1, A, v) ≈ A * v
            @test Gaius.mul_single_threaded!(u2, A, v) ≈ A * v
            @test u1 ≈ Gaius.mul(A, v)
            @test u2 ≈ Gaius.mul(A, v)
            @test u1 ≈ Gaius.mul_single_threaded(A, v)
            @test u2 ≈ Gaius.mul_single_threaded(A, v)
        end
    end
end

@time @testset "ComplexF64 Matrix-Vector $(testset_name_suffix)" begin
    for n ∈ n_values_2
        for m ∈ m_values_2
            u1 = zeros(ComplexF64, n)   |> StructArray
            u2 = zeros(ComplexF64, n)   |> StructArray
            A = rand(ComplexF64, n, m) |> StructArray
            v = rand(ComplexF64, m)    |> StructArray
            @test Gaius.mul!(u1, A, v) ≈ collect(A) * collect(v)
            @test Gaius.mul_single_threaded!(u2, A, v) ≈ collect(A) * collect(v)
            @test u1 ≈ Gaius.mul(A, v)
            @test u2 ≈ Gaius.mul(A, v)
            @test u1 ≈ Gaius.mul_single_threaded(A, v)
            @test u2 ≈ Gaius.mul_single_threaded(A, v)
        end
    end
end

@time @testset "Float32 Matrix-Vector $(testset_name_suffix)" begin
    for n ∈ n_values_2
        for m ∈ m_values_2
            u1 = zeros(Float32, n)
            u2 = zeros(Float32, n)
            A = rand(Float32, n, m)
            v = rand(Float32, m)
            @test Gaius.mul!(u1, A, v) ≈ A * v
            @test Gaius.mul_single_threaded!(u2, A, v) ≈ A * v
            @test u1 ≈ Gaius.mul(A, v)
            @test u2 ≈ Gaius.mul(A, v)
            @test u1 ≈ Gaius.mul_single_threaded(A, v)
            @test u2 ≈ Gaius.mul_single_threaded(A, v)
        end
    end
end

@time @testset "Int64 Matrix-Vector $(testset_name_suffix)" begin
    for n ∈ n_values_2
        for m ∈ m_values_2
            u1 = zeros(Int, n)
            u2 = zeros(Int, n)
            A = rand(-20:20, n, m)
            v = rand(-20:20, m)
            @test Gaius.mul!(u1, A, v) == A * v
            @test Gaius.mul_single_threaded!(u2, A, v) == A * v
            @test u1 == Gaius.mul(A, v)
            @test u2 == Gaius.mul(A, v)
            @test u1 == Gaius.mul_single_threaded(A, v)
            @test u2 == Gaius.mul_single_threaded(A, v)
        end
    end
end

@time @testset "ComplexInt32 Matrix-Vector $(testset_name_suffix)" begin
    for n ∈ n_values_2
        for m ∈ m_values_2
            u1 = zeros(Complex{Int32}, n) |> StructArray
            u2 = zeros(Complex{Int32}, n) |> StructArray
            A = StructArray{Complex{Int32}}((rand(Int32.(-10:10), n, m), rand(Int32.(-10:10), n, m)))
            v = StructArray{Complex{Int32}}((rand(Int32.(-10:10),    m), rand(Int32.(-10:10),    m)))
            @test Gaius.mul!(u1, A, v) == collect(A) * collect(v)
            @test Gaius.mul_single_threaded!(u2, A, v) == collect(A) * collect(v)
            @test u1 == Gaius.mul(A, v)
            @test u2 == Gaius.mul(A, v)
            @test u1 == Gaius.mul_single_threaded(A, v)
            @test u2 == Gaius.mul_single_threaded(A, v)
        end
    end
end

@time @testset "Float64 CoVector-Matrix $(testset_name_suffix)" begin
    for n ∈ n_values_2
        for m ∈ m_values_2
            u1 = zeros(m)
            u2 = zeros(m)
            A = rand(n, m)
            v = rand(n)
            @test Gaius.mul!(u1', v', A) ≈ v' * A
            @test Gaius.mul_single_threaded!(u2', v', A) ≈ v' * A
            @test u1' ≈ Gaius.mul(v', A)
            @test u2' ≈ Gaius.mul(v', A)
            @test u1' ≈ Gaius.mul_single_threaded(v', A)
            @test u2' ≈ Gaius.mul_single_threaded(v', A)

            @test Gaius.mul!(transpose(u1), transpose(v), A) ≈ transpose(v) * A
            @test Gaius.mul_single_threaded!(transpose(u2), transpose(v), A) ≈ transpose(v) * A
            @test transpose(u1) ≈ Gaius.mul(transpose(v), A)
            @test transpose(u2) ≈ Gaius.mul(transpose(v), A)
            @test transpose(u1) ≈ Gaius.mul_single_threaded(transpose(v), A)
            @test transpose(u2) ≈ Gaius.mul_single_threaded(transpose(v), A)
        end
    end
end

@time @testset "ComplexF32 CoVector-Matrix $(testset_name_suffix)" begin
    for n ∈ n_values_2
        for m ∈ m_values_2
            u1 = zeros(Complex{Float32}, m)   |> StructArray
            u2 = zeros(Complex{Float32}, m)   |> StructArray
            A = rand(Complex{Float32}, n, m) |> StructArray
            v = rand(Complex{Float32}, n)    |> StructArray
            @test Gaius.mul!(u1', v', A) ≈ v' * A
            @test Gaius.mul_single_threaded!(u2', v', A) ≈ v' * A
            @test u1' ≈ Gaius.mul(v', A)
            @test u2' ≈ Gaius.mul(v', A)
            @test u1' ≈ Gaius.mul_single_threaded(v', A)
            @test u2' ≈ Gaius.mul_single_threaded(v', A)

            @test Gaius.mul!(transpose(u1), transpose(v), A) ≈ transpose(v) * A
            @test Gaius.mul_single_threaded!(transpose(u2), transpose(v), A) ≈ transpose(v) * A
            @test transpose(u1) ≈ Gaius.mul(transpose(v), A)
            @test transpose(u2) ≈ Gaius.mul(transpose(v), A)
            @test transpose(u1) ≈ Gaius.mul_single_threaded(transpose(v), A)
            @test transpose(u2) ≈ Gaius.mul_single_threaded(transpose(v), A)
        end
    end
end

@time @testset "ComplexFloat64 Matrix-Matrix $(testset_name_suffix)" begin
    for sz ∈ sz_values
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m) $(testset_name_suffix)" begin
            C0 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m)))
            C1 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m)))
            C2 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m)))
            A  = StructArray{ComplexF64}((randn(n, k), randn(n, k)))
            B  = StructArray{ComplexF64}((randn(k, m), randn(k, m)))

            LinearAlgebra.mul!(C0, A, B)
            Gaius.mul!(C1, A, B)
            Gaius.mul_single_threaded!(C2, A, B)

            @test C0 ≈ C1
            @test C0 ≈ C2
            @test C1 ≈ C2

            @test Gaius.mul_single_threaded(A, B) ≈ C0
        end
    end
    for sz ∈ sz_values
        n, k, m = shuffle([sz + rand(-5:5), sz + rand(-5:5), 10])
        @testset "($n × $k) × ($k × $m) $(testset_name_suffix)" begin
            C0 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m)))
            C1 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m)))
            C2 = StructArray{ComplexF64}((zeros(n, m), zeros(n, m)))
            A  = StructArray{ComplexF64}((randn(n, k), randn(n, k)))
            B  = StructArray{ComplexF64}((randn(k, m), randn(k, m)))

            LinearAlgebra.mul!(C0, A, B)
            Gaius.mul!(C1, A, B)
            Gaius.mul_single_threaded!(C2, A, B)

            @test C0 ≈ C1
            @test C0 ≈ C2
            @test C1 ≈ C2

            @test Gaius.mul_single_threaded(A, B) ≈ C0

            @test Gaius.mul(A, B) ≈ C0
        end
    end
end

@time @testset "Float64 Matrix-Matrix $(testset_name_suffix)" begin
    for sz ∈ sz_values
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m) $(testset_name_suffix)" begin
            C0 = zeros(n, m)
            C1 = zeros(n, m)
            C2 = zeros(n, m)
            A  = randn(n, k)
            B  = randn(k, m)
            At = copy(A')
            Bt = copy(B')

            LinearAlgebra.mul!(C0, A, B)
            Gaius.mul!(C1, A, B)
            Gaius.mul_single_threaded!(C2, A, B)

            @test C0 ≈ C1
            @test C0 ≈ C2
            @test C1 ≈ C2

            @test Gaius.mul_single_threaded(A, B) ≈ C0

            fill!(C1, NaN); @test Gaius.mul!(C1, At', B)   ≈ C0
            fill!(C1, NaN); @test Gaius.mul!(C1, A,   Bt') ≈ C0
            fill!(C1, NaN); @test Gaius.mul!(C1, At', Bt') ≈ C0

            fill!(C2, NaN); @test Gaius.mul_single_threaded!(C2, At', B)   ≈ C0
            fill!(C2, NaN); @test Gaius.mul_single_threaded!(C2, A,   Bt') ≈ C0
            fill!(C2, NaN); @test Gaius.mul_single_threaded!(C2, At', Bt') ≈ C0
        end
    end
    for sz ∈ sz_values
        n, k, m = shuffle([sz + rand(-5:5), sz + rand(-5:5), 10])
        @testset "($n × $k) × ($k × $m) $(testset_name_suffix)" begin
            C0 = zeros(n, m)
            C1 = zeros(n, m)
            C2 = zeros(n, m)
            A  = randn(n, k)
            B  = randn(k, m)
            At = copy(A')
            Bt = copy(B')

            LinearAlgebra.mul!(C0, A, B)
            Gaius.mul!(C1, A, B)
            Gaius.mul_single_threaded!(C2, A, B)

            @test C0 ≈ C1
            @test C0 ≈ C2
            @test C1 ≈ C2

            @test Gaius.mul_single_threaded(A, B) ≈ C0

            fill!(C1, NaN); @test Gaius.mul!(C1, At', B)   ≈ C0
            fill!(C1, NaN); @test Gaius.mul!(C1, A,   Bt') ≈ C0
            fill!(C1, NaN); @test Gaius.mul!(C1, At', Bt') ≈ C0

            fill!(C2, NaN); @test Gaius.mul_single_threaded!(C2, At', B)   ≈ C0
            fill!(C2, NaN); @test Gaius.mul_single_threaded!(C2, A,   Bt') ≈ C0
            fill!(C2, NaN); @test Gaius.mul_single_threaded!(C2, At', Bt') ≈ C0
        end
    end
end

@time @testset "Float32 Matrix-Matrix $(testset_name_suffix)" begin
    for sz ∈ sz_values
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m) $(testset_name_suffix)" begin
            C0 = zeros(Float32, n, m)
            C1 = zeros(Float32, n, m)
            C2 = zeros(Float32, n, m)
            A  = randn(Float32, n, k)
            B  = randn(Float32, k, m)
            At = copy(A')
            Bt = copy(B')

            LinearAlgebra.mul!(C0, A, B)
            Gaius.mul!(C1, A, B)
            Gaius.mul_single_threaded!(C2, A, B)

            @test C0 ≈ C1
            @test C0 ≈ C2
            @test C1 ≈ C2

            @test Gaius.mul_single_threaded(A, B) ≈ C0

            fill!(C1, NaN32); @test Gaius.mul!(C1, At', B)   ≈ C0
            fill!(C1, NaN32); @test Gaius.mul!(C1, A,   Bt') ≈ C0
            fill!(C1, NaN32); @test Gaius.mul!(C1, At', Bt') ≈ C0

            fill!(C2, NaN32); @test Gaius.mul_single_threaded!(C2, At', B)   ≈ C0
            fill!(C2, NaN32); @test Gaius.mul_single_threaded!(C2, A,   Bt') ≈ C0
            fill!(C2, NaN32); @test Gaius.mul_single_threaded!(C2, At', Bt') ≈ C0
        end
    end
end

@time @testset "Int64 Matrix-Matrix $(testset_name_suffix)" begin
    for sz ∈ sz_values
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m) $(testset_name_suffix)" begin
            C0 = zeros(Int64, n, m)
            C1 = zeros(Int64, n, m)
            C2 = zeros(Int64, n, m)
            A  = rand(Int64.(-100:100), n, k)
            B  = rand(Int64.(-100:100), k, m)
            At = copy(A')
            Bt = copy(B')

            LinearAlgebra.mul!(C0, A, B)
            Gaius.mul!(C1, A, B)
            Gaius.mul_single_threaded!(C2, A, B)

            @test C0 == C1
            @test C0 == C2
            @test C1 == C2


            @test Gaius.mul_single_threaded(A, B) == C0

            fill!(C1, typemax(Int64)); @test Gaius.mul!(C1, At', B)   == C0
            fill!(C1, typemax(Int64)); @test Gaius.mul!(C1, A,   Bt') == C0
            fill!(C1, typemax(Int64)); @test Gaius.mul!(C1, At', Bt') == C0

            fill!(C2, typemax(Int64)); @test Gaius.mul_single_threaded!(C2, At', B)   == C0
            fill!(C2, typemax(Int64)); @test Gaius.mul_single_threaded!(C2, A,   Bt') == C0
            fill!(C2, typemax(Int64)); @test Gaius.mul_single_threaded!(C2, At', Bt') == C0

            @test Gaius.mul(A,   B) == C0
            @test Gaius.mul(At', B) == C0
            @test Gaius.mul(A,   Bt') == C0
            @test Gaius.mul(At', Bt') == C0
        end
    end
end

@time @testset "Int32 Matrix-Matrix $(testset_name_suffix)" begin
    for sz ∈ sz_values
        n, k, m = (sz .+ rand(-5:5, 3))
        @testset "($n × $k) × ($k × $m) $(testset_name_suffix)" begin
            C0 = zeros(Int32, n, m)
            C1 = zeros(Int32, n, m)
            C2 = zeros(Int32, n, m)
            A  = rand(Int32.(-100:100), n, k)
            B  = rand(Int32.(-100:100), k, m)
            At = copy(A')
            Bt = copy(B')

            LinearAlgebra.mul!(C0, A, B)
            Gaius.mul!(C1, A, B)
            Gaius.mul_single_threaded!(C2, A, B)

            @test C0 == C1
            @test C0 == C2
            @test C1 == C2


            @test Gaius.mul_single_threaded(A, B) == C0

            fill!(C1, typemax(Int32)); @test Gaius.mul!(C1, At', B)   == C0
            fill!(C1, typemax(Int32)); @test Gaius.mul!(C1, A,   Bt') == C0
            fill!(C1, typemax(Int32)); @test Gaius.mul!(C1, At', Bt') == C0

            fill!(C2, typemax(Int32)); @test Gaius.mul_single_threaded!(C2, At', B)   == C0
            fill!(C2, typemax(Int32)); @test Gaius.mul_single_threaded!(C2, A,   Bt') == C0
            fill!(C2, typemax(Int32)); @test Gaius.mul_single_threaded!(C2, At', Bt') == C0
        end
    end
end
