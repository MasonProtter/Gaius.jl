module Gaius

using LoopVectorization: @avx, VectorizationBase.AbstractStridedPointer, VectorizationBase.gesp, VectorizationBase.vload, VectorizationBase.vstore!, VectorizationBase.AVX512F
import LoopVectorization: @avx, VectorizationBase.stridedpointer
using StructArrays: StructArray
using LinearAlgebra: LinearAlgebra, Adjoint, Transpose

using UnsafeArrays: UnsafeArrays, @uviews, UnsafeArray

export blocked_mul, blocked_mul!

const DEFAULT_BLOCK_SIZE = AVX512F ? 96 : 64

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

include("pointermatrix.jl")

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


include("matmul.jl")
include("block_operations.jl")
include("kernels.jl")



end # module
