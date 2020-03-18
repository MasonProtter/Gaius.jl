
PtrArray(M::AbstractMatrix) = PtrMatrix(M)
PtrArray(v::AbstractVector) = PtrVector(v)


struct PointerMatrix{T,P <: AbstractStridedPointer} <: AbstractMatrix{T}
    ptr::P
    size::Tuple{Int,Int}
    PointerMatrix(ptr::P, size::Tuple{Int,Int}) where {T, P <: AbstractStridedPointer{T}} = new{T,P}(ptr, size)
end
@inline PtrMatrix(A::AbstractMatrix) = PointerMatrix(stridedpointer(A), size(A))
@inline Base.pointer(A::PointerMatrix) = pointer(A.ptr)
@inline Base.size(A::PointerMatrix) = A.size
@inline Base.strides(A::PointerMatrix) = strides(A.ptr)
@inline stridedpointer(A::PointerMatrix) = A.ptr

@inline Base.maybeview(A::PointerMatrix, r::UnitRange, c::UnitRange) = PointerMatrix(gesp(A.ptr, (first(r) - 1, (first(c) - 1))), (length(r), length(c)))
# getindex is important for the sake of printing the AbstractPointerMatrix. If we call something a Matrix, it's nice to support the interface if possible.
Base.@propagate_inbounds function Base.getindex(A::PointerMatrix, i::Integer, j::Integer)
    @boundscheck begin
        M, N = size(A)
        (M < i || N < j) && throw(BoundsError(A, (i,j)))
    end
    vload(stridedpointer(A), (i-1, j-1))
end
Base.@propagate_inbounds function Base.getindex(A::PointerMatrix, v, i::Integer, j::Integer)
    @boundscheck begin
        M, N = size(A)
        (M < i || N < j) && throw(BoundsError(A, (i,j)))
    end
    vstore!(stridedpointer(A), v, (i-1, j-1))
end
Base.IndexStyle(::Type{<:PointerMatrix}) = IndexCartesian()




struct PointerVector{T,P <: AbstractStridedPointer} <: AbstractVector{T}
    ptr::P
    length::Int
    PointerVector(ptr::P, length::Int) where {T, P <: AbstractStridedPointer{T}} = new{T,P}(ptr, length)
end
@inline PtrVector(v::AbstractVector) = PointerVector(stridedpointer(v), length(v))
@inline Base.pointer(v::PointerVector) = pointer(v.ptr)
@inline Base.size(v::PointerVector) = (v.length,)
@inline Base.length(v::PointerVector) = v.length
@inline Base.strides(v::PointerVector) = strides(v.ptr)
@inline stridedpointer(v::PointerVector) = v.ptr

@inline function Base.maybeview(v::PointerVector, i::UnitRange)
    PointerVector(gesp(v.ptr, (first(i) - 1,)), length(i))
end

Base.IndexStyle(::Type{<:PointerVector}) = IndexLinear()
