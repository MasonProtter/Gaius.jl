
PtrArray(M::AbstractMatrix) = PtrMatrix(M)
PtrArray(v::AbstractVector) = v#PtrVector(v)
PtrArray(v::Adjoint{T, <:AbstractVector}) where {T} = v#PointerAdjointVector(v)
PtrArray(v::Transpose{T, <:AbstractVector}) where {T} = v#PointerTransposeVector(v)

struct PointerMatrix{T,P <: AbstractStridedPointer} <: DenseMatrix{T}
    ptr::P
    size::Tuple{Int,Int}
    PointerMatrix(ptr::P, size::Tuple{Int,Int}) where {T, P <: AbstractStridedPointer{T}} = new{T,P}(ptr, size)
end
@inline PtrMatrix(A::AbstractMatrix) = PointerMatrix(stridedpointer(A), size(A))
@inline Base.pointer(A::PointerMatrix) = pointer(A.ptr)
@inline Base.size(A::PointerMatrix) = A.size
@inline Base.strides(A::PointerMatrix) = strides(A.ptr)
@inline stridedpointer(A::PointerMatrix) = A.ptr

@inline Base.maybeview(A::PointerMatrix, r::UnitRange, c::UnitRange) = PointerMatrix(gesp(A.ptr, (first(r) - 1, first(c) - 1)), (length(r), length(c)))
# getindex is important for the sake of printing the AbstractPointerMatrix. If we call something a Matrix, it's nice to support the interface if possible.
Base.@propagate_inbounds function Base.getindex(A::PointerMatrix, i::Integer, j::Integer)
    @boundscheck begin
        M, N = size(A)
        (M < i || N < j) && throw(BoundsError(A, (i,j)))
    end
    vload(stridedpointer(A), (i, j))
end
Base.@propagate_inbounds function Base.setindex!(A::PointerMatrix, v, i::Integer, j::Integer)
    @boundscheck begin
        M, N = size(A)
        (M < i || N < j) && throw(BoundsError(A, (i,j)))
    end
    vstore!(stridedpointer(A), v, (i, j))
end
Base.IndexStyle(::Type{<:PointerMatrix}) = IndexCartesian()



# This machinery below might not actually be needed.

# struct PointerVector{T,P <: AbstractStridedPointer} <: AbstractVector{T}
#     ptr::P
#     length::Int
#     PointerVector(ptr::P, length::Int) where {T, P <: AbstractStridedPointer{T}} = new{T,P}(ptr, length)
# end
# @inline PtrVector(v::AbstractVector) = PointerVector(stridedpointer(v), length(v))
# @inline Base.pointer(v::PointerVector) = pointer(v.ptr)
# @inline Base.size(v::PointerVector) = (v.length,)
# @inline Base.length(v::PointerVector) = v.length
# @inline Base.strides(v::PointerVector) = strides(v.ptr)
# @inline stridedpointer(v::PointerVector) = v.ptr

# @inline function Base.maybeview(v::PointerVector, i::UnitRange)
#     PointerVector(gesp(v.ptr, (first(i) - 1,)), length(i))
# end

# Base.IndexStyle(::Type{<:PointerVector}) = IndexLinear()



# struct PointerAdjointVector{T,P <: AbstractStridedPointer} <: AbstractMatrix{T}
#     ptr::P
#     length::Int
#     PointerAdjointVector(ptr::P, length::Int) where {T, P <: AbstractStridedPointer{T}} = new{T,P}(ptr, length)
# end
# @inline function PointerAdjointVector(v::Adjoint{T, <:AbstractVector{T}}) where {T}
#     vp = parent(v)
#     if T <: Complex
#         vp = conj.(p) #this copy is unfortunate, but I'm scared of mutating the user's data
#     end
#     PointerAdjointVector(stridedpointer(vp), length(v))
# end
# @inline Base.pointer(v::PointerAdjointVector) = pointer(v.ptr)
# @inline Base.size(v::PointerAdjointVector) = (1,v.length)
# @inline Base.length(v::PointerAdjointVector) = v.length
# @inline Base.strides(v::PointerAdjointVector) = strides(v.ptr)
# @inline stridedpointer(v::PointerAdjointVector) = v.ptr

# @inline function Base.maybeview(v::PointerAdjointVector, i::UnitRange)
#     PointerVector(gesp(v.ptr, (first(i) - 1,)), length(i))
# end

# Base.IndexStyle(::Type{<:PointerAdjointVector}) = IndexLinear()



# struct PointerTransposeVector{T,P <: AbstractStridedPointer} <: AbstractMatrix{T}
#     ptr::P
#     length::Int
#     PointerTransposeVector(ptr::P, length::Int) where {T, P <: AbstractStridedPointer{T}} = new{T,P}(ptr, length)
# end
# @inline function PointerTransposeVector(v::Transpose{T, <:AbstractVector{T}}) where {T}
#     vp = parent(v)
#     PointerTransposeVector(stridedpointer(vp), length(v))
# end
# @inline Base.pointer(v::PointerTransposeVector) = pointer(v.ptr)
# @inline Base.size(v::PointerTransposeVector) = (1,v.length)
# @inline Base.length(v::PointerTransposeVector) = v.length
# @inline Base.strides(v::PointerTransposeVector) = strides(v.ptr)
# @inline stridedpointer(v::PointerTransposeVector) = v.ptr

# @inline function Base.maybeview(v::PointerTransposeVector, i::UnitRange)
#     PointerVector(gesp(v.ptr, (first(i) - 1,)), length(i))
# end

# Base.IndexStyle(::Type{<:PointerTransposeVector}) = IndexLinear()
