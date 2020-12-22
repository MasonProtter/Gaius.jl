module Gaius

import LinearAlgebra
import LoopVectorization
import StructArrays
import UnsafeArrays
import VectorizationBase

import VectorizationBase: stridedpointer

using LinearAlgebra: Adjoint, Transpose
using LoopVectorization: @avx
using StructArrays: StructArray
using UnsafeArrays: @uviews, UnsafeArray
using VectorizationBase: AVX512F, AbstractStridedPointer, gesp, vload, vstore!

export blocked_mul
export blocked_mul!

const DEFAULT_BLOCK_SIZE = AVX512F ? 96 : 64

include("macros.jl")
include("pointermatrix.jl")
include("type-aliases.jl")

include("block-operations.jl")
include("kernels.jl")
include("matmul.jl")

end # module Gaius
