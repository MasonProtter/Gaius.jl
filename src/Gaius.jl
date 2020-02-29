module Gaius

using LoopVectorization: @avx
using LinearAlgebra: LinearAlgebra

eltypes  = Union{Float64, Float32, Int64, Int32}
MatTypes = Union{Matrix{<:eltypes}, SubArray{<:eltypes, 2, <:Matrix}}

mul!(args...) = LinearAlgebra.mul!(args...)
(*)(args...)  = Base.:(*)(args...)

function (*)(A::MatTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Matrix{T}(undef, size(A,1), size(B,2))
    mul!(C, A, B)
    C
end

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
    @assert (n == a) && (m == c) && (k == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end


function mul!(C::MatTypes, A::MatTypes, B::MatTypes; size_cutoff=64÷2, sizecheck=true)
    sizecheck && check_compatible_sizes(C, A, B)
    _mul!(C, A, B, size_cutoff)
    C
end

function _mul!(C, A, B, sz)
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    
    if n >= 2sz && m >= 2sz && k >= 2sz
        block_mat_mat_mul!(C, A, B, sz)
    elseif n >= 2sz && k >= 2sz && m <  2sz
        block_mat_vec_mul!(C, A, B, sz)
    elseif n <  2sz && k >= 2sz && m >= 2sz
        block_covec_mat_mul!(C, A, B, sz)
    elseif n >= 2sz && k <  2sz && m >= 2sz
        block_vec_covec_mul!(C, A, B, sz)
    elseif n <  2sz && k >= 2sz && m <  2sz
        block_covec_vec_mul!(C, A, B, sz)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(C, A, B, sz)
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    
    if n >= 2sz && m >= 2sz && k >= 2sz
        block_mat_mat_mul_add!(C, A, B, sz)
    elseif n >= 2sz && k >= 2sz && m <  2sz
        block_mat_vec_mul_add!(C, A, B, sz)
    elseif n <  2sz && k >= 2sz && m >= 2sz
        block_covec_mat_mul_add!(C, A, B, sz)
    elseif n >= 2sz && k <  2sz && m >= 2sz
        block_vec_covec_mul_add!(C, A, B, sz)
    elseif n <  2sz && k >= 2sz && m <  2sz
        block_covec_vec_mul_add!(C, A, B, sz)
    else
        add_gemm_kernel!(C, A, B)
    end
end

# Note this does not support changing the number of threads at runtime
macro _spawn(ex)
    if Threads.nthreads() > 1
        :(Threads.@spawn $(esc(ex)))
    else
        esc(ex)
    end
end


#----------------------------------------------------------------
#----------------------------------------------------------------
# Block Matrix multiplication

@inline function block_mat_mat_mul!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            _mul!(    C11, A11, B11, sz)
            #gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        Threads.@spawn begin
            _mul!(    C12, A11, B12, sz)
            _mul_add!(C12, A12, B22, sz)
        end
        Threads.@spawn begin
            _mul!(    C21, A21, B11, sz)
            _mul_add!(C21, A22, B21, sz)
        end
        _mul!(    C22, A21, B12, sz)
        _mul_add!(C22, A22, B22, sz)
    end
end

function block_mat_vec_mul!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:sz,     1:end]; 
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end]; 
        B21 = B[sz+1:end, 1:end];
    end
    @sync begin
        Threads.@spawn begin
            _mul!(    C11, A11, B11, sz)
            #gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul!(    C21, A21, B11, sz)
        _mul_add!(C21, A22, B21, sz)
    end
end

function block_covec_mat_mul!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            _mul!(    C11, A11, B11, sz)
            #gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul!(    C12, A11, B12, sz)
        _mul_add!(C12, A12, B22, sz)
    end
end

function block_vec_covec_mul!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz];
        A21 = A[sz+1:end, 1:sz]; 

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
    end
    @sync begin
        Threads.@spawn begin
            _mul!(    C11, A11, B11, sz)
            #gemm_kernel!(C11, A11, B11)
        end
        Threads.@spawn begin
            _mul!(C12, A11, B12, sz)
        end
        Threads.@spawn begin
            _mul!(C21, A21, B11, sz)
        end
        _mul!(C22, A21, B12, sz)
    end
end

function block_covec_vec_mul!(C, A, B, sz)
    @inbounds @views begin 
        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 

        B11 = B[1:sz,     1:sz];
        B21 = B[sz+1:end, 1:sz];
    end
    #gemm_kernel!(C, A11, B11)
    _mul!(    C, A11, B11, sz)
    _mul_add!(C, A12, B21, sz)
end

#----------------------------------------------------------------
#----------------------------------------------------------------
# Block Matrix addition-multiplication

@inline function block_mat_mat_mul_add!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            #add_gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A11, B11, sz)
            _mul_add!(C11, A12, B21, sz)
        end
        Threads.@spawn begin
            _mul_add!(C12, A11, B12, sz)
            _mul_add!(C12, A12, B22, sz)
        end
        Threads.@spawn begin
            _mul_add!(C21, A21, B11, sz)
            _mul_add!(C21, A22, B21, sz)
        end
        _mul_add!(C22, A21, B12, sz)
        _mul_add!(C22, A22, B22, sz)
    end
end

function block_mat_vec_mul_add!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:sz,     1:end]; 
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end]; 
        B21 = B[sz+1:end, 1:end];
    end
    @sync begin
        Threads.@spawn begin
            #add_gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A11, B11, sz)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul_add!(C21, A21, B11, sz)
        _mul_add!(C21, A22, B21, sz)
    end
end

function block_covec_mat_mul_add!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            #add_gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A11, B11, sz)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul_add!(C12, A11, B12, sz)
        _mul_add!(C12, A12, B22, sz)
    end
end

function block_vec_covec_mul_add!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end] 
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz];
        A21 = A[sz+1:end, 1:sz]; 

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
    end
    @sync begin
        Threads.@spawn begin
            #add_gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A11, B11, sz)
        end
        Threads.@spawn begin
            _mul_add!(C12, A11, B12, sz)
        end
        Threads.@spawn begin
            _mul_add!(C21, A21, B11, sz)
        end
        _mul_add!(C22, A21, B12, sz)
    end
end

function block_covec_vec_mul_add!(C, A, B, sz)
    @inbounds @views begin 
        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end] 

        B11 = B[1:sz,     1:sz];
        B21 = B[sz+1:end, 1:sz];
    end
    
    #add_gemm_kernel!(C, A11, B11)
    _mul_add!(C, A11, B11, sz)
    _mul_add!(C, A12, B21, sz)
end


end # module
