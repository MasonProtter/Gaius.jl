import Gaius

import BenchmarkTools
import InteractiveUtils
import LinearAlgebra
import Random
import StructArrays
import Test
import VectorizationBase

using Random: shuffle
using StructArrays: StructArray
using Test: @testset, @test, @test_logs, @test_throws

include("test_suite_preamble.jl")

@info("VectorizationBase.num_cores() is $(VectorizationBase.num_cores())")

include("block_operations.jl")
include("public_mul_coverage.jl")
include("init.jl")
include("kernels.jl")
include("matmul_coverage.jl")
include("pointermatrix.jl")

if !coverage
    include("public_mul_main.jl")
    include("matmul_main.jl")
end
