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

struct Singlethreaded <: Threading
    block_size::Int64
end

struct Multithreaded <: Threading
    singlethread_size::Int64
    singlethreaded::Singlethreaded
end

get_block_size(singlethreaded::Singlethreaded) = singlethreaded.block_size
get_block_size(multithreaded::Multithreaded) = get_block_size(multithreaded.singlethreaded)
