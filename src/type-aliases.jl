Eltypes  = Union{Float64, Float32, Int64, Int32, Int16}
MatTypesC{T} = Union{Matrix{T},
                     SubArray{T, 2, <: AbstractArray},
                     PointerMatrix{T},
                     UnsafeArray{T, 2}} # C for Column Major
MatTypesR{T} = Union{Adjoint{T,<:MatTypesC{T}},
                     Transpose{T,<:MatTypesC{T}}} # R for Row Major
MatTypes{ T} = Union{MatTypesC{T}, MatTypesR{T}}

VecTypes{T}   = Union{Vector{T}, SubArray{T, 1, <:Array}}
CoVecTypes{T} = Union{Adjoint{T,   <:VecTypes{T}},
                      Transpose{T, <:VecTypes{T}}}
