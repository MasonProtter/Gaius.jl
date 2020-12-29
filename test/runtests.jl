import Gaius

import BenchmarkTools
import InteractiveUtils
import LinearAlgebra
import Random
import StructArrays
import Test

using Gaius: blocked_mul, blocked_mul!
using Random: shuffle
using StructArrays: StructArray
using Test: @testset, @test, @test_throws

include("test_suite_preamble.jl")

include("blocked_mul_coverage.jl")
include("kernels.jl")
include("matmul.jl")
include("pointermatrix.jl")

if !coverage
    include("blocked_mul.jl")
end
