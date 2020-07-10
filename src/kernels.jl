function gemm_kernel!(C, A, B)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += A[n,k] * B[k,m]
        end
        C[n,m] = Cₙₘ
    end
end

function add_gemm_kernel!(C::MatTypes, A::MatTypes, B::MatTypes)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += A[n,k] * B[k,m]
        end
        C[n,m] += Cₙₘ
    end
end

add_gemm_kernel!(C::MatTypes, A::MatTypes, B::MatTypes, ::Val{1}) = add_gemm_kernel!(C, A, B)

function add_gemm_kernel!(C::MatTypes, A::MatTypes, B::MatTypes, ::Val{-1})
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ -= A[n,k] * B[k,m]
        end
        C[n,m] += Cₙₘ
    end
end

function add_gemm_kernel!(C::MatTypes, A::MatTypes, B::MatTypes, ::Val{factor}) where {factor}
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += factor * A[n,k] * B[k,m]
        end
        C[n,m] += Cₙₘ
    end
end

#____________

function gemm_kernel!(u::VecTypes, A::MatTypes, v::VecTypes)
    @avx for n ∈ 1:size(A, 1)
        uₙ = zero(eltype(u))
        for k ∈ 1:size(A, 2)
            uₙ += A[n,k] * v[k]
        end
        u[n] = uₙ
    end
end

function add_gemm_kernel!(u::VecTypes, A::MatTypes, v::VecTypes)
    @avx for n ∈ 1:size(A, 1)
        uₙ = zero(eltype(u))
        for k ∈ 1:size(A, 2)
            uₙ += A[n,k] * v[k]
        end
        u[n] += uₙ
    end
end

function add_gemm_kernel!(u::VecTypes, A::MatTypes, v::VecTypes, ::Val{-1})
    @avx for n ∈ 1:size(A, 1)
        uₙ = zero(eltype(u))
        for k ∈ 1:size(A, 2)
            uₙ -= A[n,k] * v[k]
        end
        u[n] += uₙ
    end
end

function add_gemm_kernel!(u::VecTypes, A::MatTypes, v::VecTypes, ::Val{factor}) where {factor}
    @avx for n ∈ 1:size(A, 1)
        uₙ = zero(eltype(u))
        for k ∈ 1:size(A, 2)
            uₙ += factor * A[n,k] * v[k]
        end
        u[n] += uₙ
    end
end

#____________

function gemm_kernel!(u::CoVecTypes, v::CoVecTypes, A::MatTypes)
    @avx for m ∈ 1:size(A, 2)
        uₘ = zero(eltype(u))
        for k ∈ 1:size(A, 1)
            uₘ += v[k] * A[k, m]
        end
        u[m] = uₘ
    end
end

function add_gemm_kernel!(u::CoVecTypes, v::CoVecTypes, A::MatTypes)
    @avx for m ∈ 1:size(A, 2)
        uₘ = zero(eltype(u))
        for k ∈ 1:size(A, 1)
            uₘ += v[k] * A[k, m]
        end
        u[m] += uₘ
    end
end

function add_gemm_kernel!(u::CoVecTypes, v::CoVecTypes, A::MatTypes, ::Val{-1})
    @avx for m ∈ 1:size(A, 2)
        uₘ = zero(eltype(u))
        for k ∈ 1:size(A, 1)
            uₘ -= v[k] * A[k, m]
        end
        u[m] += uₘ
    end
end

function add_gemm_kernel!(u::CoVecTypes, v::CoVecTypes, A::MatTypes, ::Val{factor}) where {factor}
    @avx for m ∈ 1:size(A, 2)
        uₘ = zero(eltype(u))
        for k ∈ 1:size(A, 1)
            uₘ += factor * v[k] * A[k, m]
        end
        u[m] += uₘ
    end
end
