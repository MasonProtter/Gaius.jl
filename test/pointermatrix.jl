@time @testset "PointerMatrix" begin
    m = rand(Float64, 10, 20)
    block = Gaius.PtrArray(m)
    @test Base.pointer(block) == Base.pointer(block.ptr)
    @test Base.strides(block) == Base.strides(block.ptr)
    block[1] = 2.3
    @test block[1] == 2.3
    block[4, 5] = 67.89
    @test block[4, 5] == 67.89
end
