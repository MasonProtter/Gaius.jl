import Gaius

import BenchmarkTools
import InteractiveUtils
import LinearAlgebra
import Random
import StructArrays
import Test

using Random: shuffle
using StructArrays: StructArray
using Test: @testset, @test, @test_throws

include("test_suite_preamble.jl")

include("public_mul_coverage.jl")
include("kernels.jl")
include("matmul.jl")
include("pointermatrix.jl")

if !coverage
    include("public_mul.jl")
end
