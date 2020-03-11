module Gaius

using LoopVectorization: @avx, VectorizationBase.AbstractStridedPointer, VectorizationBase.gesp, VectorizationBase.vload, VectorizationBase.vstore!, VectorizationBase.AVX512F
import LoopVectorization: @avx, VectorizationBase.stridedpointer
using StructArrays: StructArray
using LinearAlgebra: LinearAlgebra

export blocked_mul, blocked_mul!

const DEFAULT_BLOCK_SIZE = AVX512F ? 96 : 64

const Eltypes  = Union{Float64, Float32, Int64, Int32, Int16}
const MatTypesC{T <: Eltypes} = Union{Matrix{T}, SubArray{T, 2, <: Array}} # C for Column Major
const MatTypesR{T <: Eltypes} = Union{LinearAlgebra.Adjoint{T,<:MatTypesC{T}}, LinearAlgebra.Transpose{T,<:MatTypesC{T}}} # R for Row Major
const MatTypes{T <: Eltypes} = Union{MatTypesC{T}, MatTypesR{T}}

const   VecTypes{T <: Eltypes} = Union{Vector{T}, SubArray{T, 1, <:Array}}
const CoVecTypes{T <: Eltypes} = Union{LinearAlgebra.Adjoint{T,<:VecTypes{T}}, LinearAlgebra.Transpose{T, <:VecTypes{T}}}

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
include("matmul.jl")
include("block_operations.jl")
include("kernels.jl")



end # module
