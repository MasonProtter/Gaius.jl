module Gaius

using LoopVectorization: @avx, VectorizationBase.PackedStridedPointer, VectorizationBase.SparseStridedPointer, VectorizationBase.gep, VectorizationBase.vload, VectorizationBase.vstore!, VectorizationBase.REGISTER_SIZE
import LoopVectorization: @avx, VectorizationBase.stridedpointer
using StructArrays: StructArray
using LinearAlgebra: LinearAlgebra


const DEFAULT_BLOCK_SIZE = REGISTER_SIZE == 64 ? 128 : 104
const Eltypes  = Union{Float64, Float32, Int64, Int32, Int16}
const MatTypes{T <: Eltypes} = Union{Matrix{T}, SubArray{T, 2, <: Array}}

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
