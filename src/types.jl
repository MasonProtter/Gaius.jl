const Eltypes       = Union{Float64, Float32, Int64, Int32, Int16}
const MatTypesC{T}  = Union{Matrix{T},
                            SubArray{T, 2, <: AbstractArray},
                            UnsafeArray{T, 2}} # C for Column Major
const MatTypesR{T}  = Union{Adjoint{T,<:MatTypesC{T}},
                            Transpose{T,<:MatTypesC{T}}} # R for Row Major
const MatTypes{T}   = Union{MatTypesC{T}, MatTypesR{T}}
const VecTypes{T}   = Union{Vector{T}, SubArray{T, 1, <:Array}}
const CoVecTypes{T} = Union{Adjoint{T,   <:VecTypes{T}},
                            Transpose{T, <:VecTypes{T}}}

abstract type Threading end

struct Multithreaded <: Threading
    singlethread_size::Int
    # TODO: Add `block_size`
end

struct Singlethreaded <: Threading
end

const singlethreaded = Singlethreaded()
