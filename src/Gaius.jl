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

export t_blocked_mul
export t_blocked_mul!

include("global_constants.jl")
include("types.jl")

include("block_operations.jl")
include("check_compatible_sizes.jl")
include("choose_block_size.jl")
include("kernels.jl")
include("matmul.jl")
include("pointermatrix.jl")
include("public_mul.jl")

end # module Gaius
