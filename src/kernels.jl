function gemm_kernel!(C, A, B)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        C_n_m = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            C_n_m += A[n,k] * B[k,m]
        end
        C[n,m] = C_n_m
    end
end

function add_gemm_kernel!(C::MatTypes, A::MatTypes, B::MatTypes)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        C_n_m = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            C_n_m += A[n,k] * B[k,m]
        end
        C[n,m] += C_n_m
    end
end

add_gemm_kernel!(C::MatTypes, A::MatTypes, B::MatTypes, ::Val{1}) = add_gemm_kernel!(C, A, B)

function add_gemm_kernel!(C::MatTypes, A::MatTypes, B::MatTypes, ::Val{-1})
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        C_n_m = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            C_n_m -= A[n,k] * B[k,m]
        end
        C[n,m] += C_n_m
    end
end

function add_gemm_kernel!(C::MatTypes, A::MatTypes, B::MatTypes, ::Val{factor}) where {factor}
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        C_n_m = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            C_n_m += factor * A[n,k] * B[k,m]
        end
        C[n,m] += C_n_m
    end
end

#____________

function gemm_kernel!(u::VecTypes, A::MatTypes, v::VecTypes)
    @avx for n ∈ 1:size(A, 1)
        u_n = zero(eltype(u))
        for k ∈ 1:size(A, 2)
            u_n += A[n,k] * v[k]
        end
        u[n] = u_n
    end
end

function add_gemm_kernel!(u::VecTypes, A::MatTypes, v::VecTypes)
    @avx for n ∈ 1:size(A, 1)
        u_n = zero(eltype(u))
        for k ∈ 1:size(A, 2)
            u_n += A[n,k] * v[k]
        end
        u[n] += u_n
    end
end

function add_gemm_kernel!(u::VecTypes, A::MatTypes, v::VecTypes, ::Val{-1})
    @avx for n ∈ 1:size(A, 1)
        u_n = zero(eltype(u))
        for k ∈ 1:size(A, 2)
            u_n -= A[n,k] * v[k]
        end
        u[n] += u_n
    end
end

function add_gemm_kernel!(u::VecTypes, A::MatTypes, v::VecTypes, ::Val{factor}) where {factor}
    @avx for n ∈ 1:size(A, 1)
        u_n = zero(eltype(u))
        for k ∈ 1:size(A, 2)
            u_n += factor * A[n,k] * v[k]
        end
        u[n] += u_n
    end
end

#____________

function gemm_kernel!(u::CoVecTypes, v::CoVecTypes, A::MatTypes)
    @avx for m ∈ 1:size(A, 2)
        u_m = zero(eltype(u))
        for k ∈ 1:size(A, 1)
            u_m += v[k] * A[k, m]
        end
        u[m] = u_m
    end
end

function add_gemm_kernel!(u::CoVecTypes, v::CoVecTypes, A::MatTypes)
    @avx for m ∈ 1:size(A, 2)
        u_m = zero(eltype(u))
        for k ∈ 1:size(A, 1)
            u_m += v[k] * A[k, m]
        end
        u[m] += u_m
    end
end

function add_gemm_kernel!(u::CoVecTypes, v::CoVecTypes, A::MatTypes, ::Val{-1})
    @avx for m ∈ 1:size(A, 2)
        u_m = zero(eltype(u))
        for k ∈ 1:size(A, 1)
            u_m -= v[k] * A[k, m]
        end
        u[m] += u_m
    end
end

function add_gemm_kernel!(u::CoVecTypes, v::CoVecTypes, A::MatTypes, ::Val{factor}) where {factor}
    @avx for m ∈ 1:size(A, 2)
        u_m = zero(eltype(u))
        for k ∈ 1:size(A, 1)
            u_m += factor * v[k] * A[k, m]
        end
        u[m] += u_m
    end
end
