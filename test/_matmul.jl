# The following variables need to be defined before `include`-ing this file:
# `testset_name_suffix`
# `sz_values`
# `n_values`
# `k_values`
# `m_values`

@time @testset "_mul!: C, A, B $(testset_name_suffix)" begin
    multithreaded = Gaius.Multithreaded(2^24)
    for threading in [Gaius.singlethreaded, multithreaded]
        for sz in sz_values
            for n in n_values
                for k in k_values
                    for m in m_values
                        A = randn(Float64, n, k)
                        B = randn(Float64, k, m)
                        C1 = zeros(Float64, n, m)
                        C2 = zeros(Float64, n, m)
                        Gaius._mul!(    threading, C1, A, B, sz)
                        Gaius._mul_add!(threading, C2, A, B, sz, Val(1))
                        @test A * B ≈ C1
                        @test A * B ≈ C2
                    end
                end
            end
        end
    end
end

@time @testset "_mul!: C::VecTypes, A::MatTypes, B::VecTypes $(testset_name_suffix)" begin
    multithreaded = Gaius.Multithreaded(2^24)
    for threading in [Gaius.singlethreaded, multithreaded]
        for sz in sz_values
            for n in n_values
                for k in k_values
                    for m in m_values
                        A = randn(Float64, n, k)
                        B = randn(Float64, k)
                        C1 = zeros(Float64, n)
                        C2 = zeros(Float64, n)
                        Gaius._mul!(    threading, C1, A, B, sz)
                        Gaius._mul_add!(threading, C2, A, B, sz, Val(1))
                        @test A * B ≈ C1
                        @test A * B ≈ C2
                    end
                end
            end
        end
    end
end

@time @testset "_mul!: C::CoVecTypes, A::CoVecTypes, B::MatTypes $(testset_name_suffix)" begin
    multithreaded = Gaius.Multithreaded(2^24)
    for threading in [Gaius.singlethreaded, multithreaded]
        for sz in sz_values
            for n in n_values
                for k in k_values
                    for m in m_values
                        A = adjoint(randn(Float64, n))
                        B = randn(Float64, n, m)
                        C1 = adjoint(zeros(Float64, m))
                        C2 = adjoint(zeros(Float64, m))
                        Gaius._mul!(    threading, C1, A, B, sz)
                        Gaius._mul_add!(threading, C2, A, B, sz, Val(1))
                        @test A * B ≈ C1
                        @test A * B ≈ C2
                    end
                end
            end
        end
    end
end
