module Gaius

using LoopVectorization: @avx, VectorizationBase.PackedStridedPointer, VectorizationBase.SparseStridedPointer, VectorizationBase.gep, VectorizationBase.vload, VectorizationBase.vstore!, VectorizationBase.REGISTER_SIZE
import LoopVectorization: @avx, VectorizationBase.stridedpointer
using LinearAlgebra: LinearAlgebra


const DEFAULT_BLOCK_SIZE = REGISTER_SIZE == 64 ? 128 : 104
const Eltypes  = Union{Float64, Float32, Int64, Int32, Int16}
const MatTypes{T <: Eltypes} = Union{Matrix{T}, SubArray{T, 2, <: Array}}

mul!(args...) = LinearAlgebra.mul!(args...)
(*)(args...)  = Base.:(*)(args...)

function (*)(A::MatTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Matrix{T}(undef, size(A,1), size(B,2))
    mul!(C, A, B)
    C
end

function check_compatible_sizes(C, A, B)
    n, m = size(C)
    a, k = size(A)
    b, c = size(B)
    @assert (n == a) && (m == c) && (k == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end

abstract type AbstractPointerMatrix{T} <: AbstractMatrix{T} end
struct PointerMatrix{T} <: AbstractPointerMatrix{T}
    ptr::Ptr{T}
    size::NTuple{2,Int}
    stride2::Int
end
struct SparseStridePointerMatrix{T} <: AbstractPointerMatrix{T}
    ptr::Ptr{T}
    size::Tuple{Int,Int}
    strides::Tuple{Int,Int}
end
@inline PtrMatrix(A::MatTypes) = PointerMatrix(pointer(A), size(A), stride(A, 2))
@inline PtrMatrix(A::SubArray{T,2,<:Array{T,<:Any},<:Tuple{Int64,Vararg}}) where {T <: Eltypes} = SparseStridePointerMatrix(pointer(A), size(A), strides(A))
@inline Base.pointer(A::AbstractPointerMatrix) = A.ptr
@inline Base.size(A::AbstractPointerMatrix) = A.size
@inline Base.strides(A::PointerMatrix) = (1, A.stride2)
@inline Base.strides(A::SparseStridePointerMatrix) = A.strides
@inline stridedpointer(A::PointerMatrix) = PackedStridedPointer(A.ptr, (A.stride2,))
@inline stridedpointer(A::SparseStridePointerMatrix) = SparseStridedPointer(A.ptr, A.strides)
@inline Base.maybeview(A::PointerMatrix, r::UnitRange, c::UnitRange) = PointerMatrix(gep(pointer(A), first(r) - 1 + (first(c) - 1)*A.stride2), (length(r), length(c)), A.stride2)
@inline Base.maybeview(A::SparseStridePointerMatrix, r::UnitRange, c::UnitRange) = @inbounds SparseStridePointerMatrix(gep(pointer(A), (first(r) - 1)*A.strides[1] + (first(c) - 1)*A.strides[2]), (length(r), length(c)), A.strides)
# getindex is important for the sake of printing the AbstractPointerMatrix. If we call something a Matrix, it's nice to support the interface if possible.
Base.@propagate_inbounds function Base.getindex(A::AbstractPointerMatrix, i::Integer, j::Integer)
    @boundscheck begin
        M, N = size(A)
        (M < i || N < j) && throw(BoundsError(A, (i,j)))
    end
    vload(stridedpointer(A), (i-1, j-1))
end
Base.@propagate_inbounds function Base.getindex(A::AbstractPointerMatrix, v, i::Integer, j::Integer)
    @boundscheck begin
        M, N = size(A)
        (M < i || N < j) && throw(BoundsError(A, (i,j)))
    end
    vstore!(stridedpointer(A), v, (i-1, j-1))
end
Base.IndexStyle(::Type{<:AbstractPointerMatrix}) = IndexCartesian()


function mul!(C::MatTypes, A::MatTypes, B::MatTypes; block_size = DEFAULT_BLOCK_SIZE, sizecheck=true)
    sizecheck && check_compatible_sizes(C, A, B)
    GC.@preserve C A B _mul!(PtrMatrix(C), PtrMatrix(A), PtrMatrix(B), block_size >>> 1)
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
        esc(Expr(:macrocall, Expr(:(.), :Threads, QuoteNode(Symbol("@spawn"))), __source__, ex))
    else
        esc(ex)
    end
end
macro _sync(ex)
    if Threads.nthreads() > 1
        esc(Expr(:macrocall, Symbol("@sync"), __source__, ex))
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
    @_sync begin
        @_spawn begin
            #_mul!(    C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        @_spawn begin
            _mul!(    C12, A11, B12, sz)
            _mul_add!(C12, A12, B22, sz)
        end
        @_spawn begin
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
    @_sync begin
        @_spawn begin
            #_mul!(    C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul!(    C21, A21, B11, sz)
        _mul_add!(C21, A22, B21, sz)
    end
end

function block_covec_mat_mul!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:end,    1:sz]; C12 = C[1:end,    sz+1:end] 

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end] 

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @_sync begin
        @_spawn begin
            #_mul!(    C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
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

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end]; 

        B11 = B[1:end,     1:sz]; B12 = B[1:end,     sz+1:end] 
    end
    @_sync begin
        @_spawn begin
            #_mul!(    C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
        end
        @_spawn begin
            _mul!(C12, A11, B12, sz)
        end
        @_spawn begin
            _mul!(C21, A21, B11, sz)
        end
        _mul!(C22, A21, B12, sz)
    end
end

function block_covec_vec_mul!(C, A, B, sz)
    @inbounds @views begin 
        A11 = A[1:end,    1:sz]; A12 = A[1:sz,     sz+1:end] 

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    gemm_kernel!(C, A11, B11)
    #_mul!(    C, A11, B11, sz)
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
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11)
            #_mul_add!(C11, A11, B11, sz)
            _mul_add!(C11, A12, B21, sz)
        end
        @_spawn begin
            _mul_add!(C12, A11, B12, sz)
            _mul_add!(C12, A12, B22, sz)
        end
        @_spawn begin
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
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11)
            #_mul_add!(C11, A11, B11, sz)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul_add!(C21, A21, B11, sz)
        _mul_add!(C21, A22, B21, sz)
    end
end

function block_covec_mat_mul_add!(C, A, B, sz)
    @inbounds @views begin 
        C11 = C[1:end,    1:sz]; C12 = C[1:end,     sz+1:end] 

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end] 

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end] 
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11)
            #_mul_add!(C11, A11, B11, sz)
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

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end]; 

        B11 = B[1:end,    1:sz]; B12 = B[1:end,    sz+1:end] 
    end
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11)
            #_mul_add!(C11, A11, B11, sz)
        end
        @_spawn begin
            _mul_add!(C12, A11, B12, sz)
        end
        @_spawn begin
            _mul_add!(C21, A21, B11, sz)
        end
        _mul_add!(C22, A21, B12, sz)
    end
end

function block_covec_vec_mul_add!(C, A, B, sz)
    @inbounds @views begin 
        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end] 

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    add_gemm_kernel!(C, A11, B11)
    #_mul_add!(C, A11, B11, sz)
    _mul_add!(C, A12, B21, sz)
end

# , Val{sz}(), Val{sz}(), Val{sz}()

#----------------------------------------------------------------
#----------------------------------------------------------------
# The workhorses

function gemm_kernel!(C, A, B)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += A[n,k] * B[k,m]
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

end # module
