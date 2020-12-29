struct PointerMatrix{T,P <: AbstractStridedPointer} <: DenseMatrix{T}
    ptr::P
    size::Tuple{Int,Int}
    PointerMatrix(ptr::P, size::Tuple{Int,Int}) where {T, P <: AbstractStridedPointer{T}} = new{T,P}(ptr, size)
end

const Eltypes       = Union{Float64, Float32, Int64, Int32, Int16}
const MatTypesC{T}  = Union{Matrix{T},
                            SubArray{T, 2, <: AbstractArray},
                            PointerMatrix{T},
                            UnsafeArray{T, 2}} # C for Column Major
const MatTypesR{T}  = Union{Adjoint{T,<:MatTypesC{T}},
                            Transpose{T,<:MatTypesC{T}}} # R for Row Major
const MatTypes{T}   = Union{MatTypesC{T}, MatTypesR{T}}
const VecTypes{T}   = Union{Vector{T}, SubArray{T, 1, <:Array}}
const CoVecTypes{T} = Union{Adjoint{T,   <:VecTypes{T}},
                            Transpose{T, <:VecTypes{T}}}
