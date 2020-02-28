module Gaius

using LoopVectorization: @avx
using LinearAlgebra: LinearAlgebra


mul!(args...) = LinearAlgebra.mul!(args...)
(*)(args...)  = Base.:(*)(args...)


eltypes  = Union{Float64, Float32, Int64, Int32}
MatTypes = Union{Matrix{<:eltypes}, SubArray{<:eltypes, 2, <:Matrix}}

(*)(A::MatTypes, B::MatTypes) = mul!(Matrix{promote_type(eltype(A), eltype(B))}(undef, size(A,1), size(B,2)))

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


function mul!(C::MatTypes, A::MatTypes, B::MatTypes)
    sz = 104÷2
    n, k, m = check_compatible_sizes(C, A, B)
    if n >= 2sz && m >= 2sz && k >= 2sz
        #nmid, mmid, kmid = (n, m, k) .÷ 2
        @views begin 
            C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 
            C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

            A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 
            A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

            B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
            B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
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
    elseif n >= 2sz && k >= 2sz && m <  2sz
        @views begin 
            C11 = C[1:sz,     1:end]; 
            C21 = C[sz+1:end, 1:end];

            A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 
            A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

            B11 = B[1:sz,     1:end]; 
            B21 = B[sz+1:end, 1:end];
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul!(    C11, A11, B11)
                blocked_mul_add!(C11, A12, B21)
            end
            blocked_mul!(    C21, A21, B11)
            blocked_mul_add!(C21, A22, B21)
        end
    elseif n <  2sz && k >= 2sz && m >= 2sz
        @views begin 
            C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 

            A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 

            B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
            B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul!(    C11, A11, B11)
                blocked_mul_add!(C11, A12, B21)
            end
            blocked_mul!(    C12, A11, B12)
            blocked_mul_add!(C12, A12, B22)
        end
    elseif n >= 2sz && k <  2sz && m >= 2sz
        @views begin 
            C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 
            C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

            A11 = A[1:sz,     1:sz];
            A21 = A[sz+1:end, 1:sz]; 

            B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul!(    C11, A11, B11)
            end
            Threads.@spawn begin
                blocked_mul!(    C12, A11, B12)
            end
            Threads.@spawn begin
                blocked_mul!(    C21, A21, B11)
            end
            blocked_mul!(    C22, A21, B12)
        end
    elseif n <  2sz && k >= 2sz && m <  2sz
        @views begin 
            C11 = C

            A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 

            B11 = B[1:sz,     1:sz];
            B21 = B[sz+1:end, 1:sz];
        end
        blocked_mul!(    C11, A11, B11)
        blocked_mul_add!(C11, A12, B21)
    else
        gemm_kernel!(C, A, B)
    end
end

function mul_add!(C::MatTypes, A::MatTypes, B::MatTypes)
    sz = 104÷2
    n, k, m = check_compatible_sizes(C, A, B)
    if n >= 2sz && m >= 2sz && k >= 2sz
        #nmid, mmid, kmid = (n, m, k) .÷ 2
        @views begin 
            C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 
            C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

            A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 
            A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

            B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
            B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul_add!(    C11, A11, B11)
                blocked_mul_add!(C11, A12, B21)
            end
            Threads.@spawn begin
                blocked_mul_add!(    C12, A11, B12)
                blocked_mul_add!(C12, A12, B22)
            end
            Threads.@spawn begin
                blocked_mul_add!(    C21, A21, B11)
                blocked_mul_add!(C21, A22, B21)
            end
            blocked_mul_add!(    C22, A21, B12)
            blocked_mul_add!(C22, A22, B22)
        end
    elseif n >= 2sz && k >= 2sz && m <  2sz
        @views begin 
            C11 = C[1:sz,     1:end]; 
            C21 = C[sz+1:end, 1:end];

            A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 
            A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

            B11 = B[1:sz,     1:end]; 
            B21 = B[sz+1:end, 1:end];
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul_add!(    C11, A11, B11)
                blocked_mul_add!(C11, A12, B21)
            end
            blocked_mul_add!(    C21, A21, B11)
            blocked_mul_add!(C21, A22, B21)
        end
    elseif n <  2sz && k >= 2sz && m >= 2sz
        @views begin 
            C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 

            A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 

            B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
            B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul_add!(    C11, A11, B11)
                blocked_mul_add!(C11, A12, B21)
            end
            blocked_mul_add!(    C12, A11, B12)
            blocked_mul_add!(C12, A12, B22)
        end
    elseif n >= 2sz && k <  2sz && m >= 2sz
        @views begin 
            C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 
            C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

            A11 = A[1:sz,     1:sz];
            A21 = A[sz+1:end, 1:sz]; 

            B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
        end
        @sync begin
            Threads.@spawn begin
                blocked_mul_add!(    C11, A11, B11)
            end
            Threads.@spawn begin
                blocked_mul_add!(    C12, A11, B12)
            end
            Threads.@spawn begin
                blocked_mul_add!(    C21, A21, B11)
            end
            blocked_mul_add!(    C22, A21, B12)
        end
    elseif n <  2sz && k >= 2sz && m <  2sz
        @views begin 
            C11 = C

            A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 

            B11 = B[1:sz,     1:sz];
            B21 = B[sz+1:end, 1:sz];
        end
        blocked_mul_add!(    C11, A11, B11)
        blocked_mul_add!(C11, A12, B21)
    else
        add_gemm_kernel!(C, A, B)
    end
end


end # module
