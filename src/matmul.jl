function check_compatible_sizes(C, A, B)
    n, m = size(C)
    a, k = size(A)
    b, c = size(B)
    @assert (n == a) && (m == c) && (k == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end

mul!(args...) = LinearAlgebra.mul!(args...)
(*)(args...)  = Base.:(*)(args...)

function (*)(A::MatTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Matrix{T}(undef, size(A,1), size(B,2))
    mul!(C, A, B)
    C
end

function mul!(C::MatTypes{T}, A::MatTypes{T}, B::MatTypes{T}; block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C, A, B)
    GC.@preserve C A B _mul!(PtrMatrix(C), PtrMatrix(A), PtrMatrix(B), block_size >>> 1)
    C
end

function mul!(C::StructArray{Complex{T}, 2}, A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 2};
              block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C, A, B)
    GC.@preserve C A B begin
        Cre, Cim = PtrMatrix(C.re), PtrMatrix(C.im)
        Are, Aim = PtrMatrix(A.re), PtrMatrix(A.im)
        Bre, Bim = PtrMatrix(B.re), PtrMatrix(B.im)
        
        _mul!(    Cre, Are, Bre,  block_size >>> 1)            # C.re = A.re * B.re
        _mul_add!(Cre, Aim, Bim,  block_size >>> 1, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    Cim, Are, Bim,  block_size >>> 1)            # C.im = A.re * B.im
        _mul_add!(Cim, Aim, Bre,  block_size >>> 1)            # C.im = C.im + A.im * B.re
    end
    C
end

function _mul!(C, A, B, sz)
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    
    if n >= 2sz && m >= 2sz && k >= 2sz
        block_mat_mat_mul!(C, A, B, sz)
    elseif n >= 2sz && k >= 2sz && m <  2sz
        block_mat_vec_mul!(C, A, B, sz)
    elseif n <  2sz && k >= 2sz && m >= 2sz
        block_covec_mat_mul!(C, A, B, sz)
    elseif n >= 2sz && k <  2sz && m >= 2sz
        block_vec_covec_mul!(C, A, B, sz)
    elseif n <  2sz && k >= 2sz && m <  2sz
        block_covec_vec_mul!(C, A, B, sz)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    if n >= 2sz && m >= 2sz && k >= 2sz
        block_mat_mat_mul_add!(C, A, B, sz,   Val(factor))
    elseif n >= 2sz && k >= 2sz && m <  2sz
        block_mat_vec_mul_add!(C, A, B, sz,   Val(factor))
    elseif n <  2sz && k >= 2sz && m >= 2sz
        block_covec_mat_mul_add!(C, A, B, sz, Val(factor))
    elseif n >= 2sz && k <  2sz && m >= 2sz
        block_vec_covec_mul_add!(C, A, B, sz, Val(factor))
    elseif n <  2sz && k >= 2sz && m <  2sz
        block_covec_vec_mul_add!(C, A, B, sz, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end
