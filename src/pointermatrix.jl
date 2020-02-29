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

