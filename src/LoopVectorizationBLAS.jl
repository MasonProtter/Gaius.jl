module LoopVectorizationBLAS

using LoopVectorization: @avx
#using Restacker
#using UnsafeArrays: UnsafeArrays, @uviews

(*)(args...) = Base.:(*)(args...)

export blocked_mul!

eltypes  = Union{Float64, Float32, Int64, Int32}
MatTypes = Union{Matrix{<:eltypes}, SubArray{<:eltypes, 2, <:Matrix}}

function gemm_kernel!(C, A, B)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += A[n,k] * B[k,m]
        end
        C[n,m] = Cₙₘ
    end
end

function gemm_kernel!(C, A, B, D, E)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += A[n,k] * B[k,m] + D[n,k]*E[k,n] 
        end
        C[n,m] = Cₙₘ
    end
end

function add_gemm_kernel!(C, A, B)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += A[n,k] * B[k,m]
        end
        C[n,m] += Cₙₘ
    end
end

function check_compatible_sizes(C, A, B)
    n, m = size(C)
    a, k = size(A)
    b, c = size(B)
    #@assert (n == a) && (m == c) && (k == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    n, k, m
end

const sz_lim = 104

function blocked_mul!(C::MatTypes, A::MatTypes, B::MatTypes)
    n, k, m = check_compatible_sizes(C, A, B)
    if n >= sz_lim && m >= sz_lim && k >= sz_lim
        nmid, mmid, kmid = (n, m, k) .÷ 2
        @views begin 
            C11 = C[1:nmid,     1:mmid]; C12 = C[1:nmid,     mmid+1:end] 
            C21 = C[nmid+1:end, 1:mmid]; C22 = C[nmid+1:end, mmid+1:end]

            A11 = A[1:nmid,     1:kmid]; A12 = A[1:nmid,     kmid+1:end] 
            A21 = A[nmid+1:end, 1:kmid]; A22 = A[nmid+1:end, kmid+1:end]

            B11 = B[1:kmid,     1:mmid]; B12 = B[1:kmid,     mmid+1:end] 
            B21 = B[kmid+1:end, 1:mmid]; B22 = B[kmid+1:end, mmid+1:end]
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul!(    C11, A11, B11)
                blocked_mul_add!(C11, A12, B21)
            end
            Threads.@spawn begin
                blocked_mul!(    C12, A11, B12)
                blocked_mul_add!(C12, A12, B22)
            end
            Threads.@spawn begin
                blocked_mul!(    C21, A21, B11)
                blocked_mul_add!(C21, A22, B21)
            end
            blocked_mul!(    C22, A21, B12)
            blocked_mul_add!(C22, A22, B22)
        end
    else
        gemm_kernel!(C, A, B)
    end
end

function blocked_mul_add!(C::MatTypes, A::MatTypes, B::MatTypes)
    n, k, m = check_compatible_sizes(C, A, B)
    if n >= sz_lim && m >= sz_lim && k >= sz_lim
        nmid, mmid, kmid = (n, m, k) .÷ 2
        @views begin       
            C11 = C[1:nmid,     1:mmid] ; C12 = C[1:nmid,     mmid+1:end] 
            C21 = C[nmid+1:end, 1:mmid] ; C22 = C[nmid+1:end, mmid+1:end]

            A11 = A[1:nmid,     1:kmid] ; A12 = A[1:nmid,     kmid+1:end] 
            A21 = A[nmid+1:end, 1:kmid] ; A22 = A[nmid+1:end, kmid+1:end]

            B11 = B[1:kmid,     1:mmid] ; B12 = B[1:kmid,     mmid+1:end] 
            B21 = B[kmid+1:end, 1:mmid] ; B22 = B[kmid+1:end, mmid+1:end]
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul_add!(C11, A11, B11)
                blocked_mul_add!(C11, A12, B21)
            end
            Threads.@spawn begin
                blocked_mul_add!(C12, A11, B12)
                blocked_mul_add!(C12, A12, B22)
            end
            Threads.@spawn begin
                blocked_mul_add!(C21, A21, B11)
                blocked_mul_add!(C21, A22, B21)
            end
            blocked_mul_add!(C22, A21, B12)
            blocked_mul_add!(C22, A22, B22)

        end
    else
        add_gemm_kernel!(C, A, B)
    end
end

end # module
