
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

function mul!(C::MatTypes, A::MatTypes, B::MatTypes; block_size = DEFAULT_BLOCK_SIZE, sizecheck=true)
    sizecheck && check_compatible_sizes(C, A, B)
    GC.@preserve C A B _mul!(PtrMatrix(C), PtrMatrix(A), PtrMatrix(B), block_size >>> 1)
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

function _mul_add!(C, A, B, sz)
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    
    if n >= 2sz && m >= 2sz && k >= 2sz
        block_mat_mat_mul_add!(C, A, B, sz)
    elseif n >= 2sz && k >= 2sz && m <  2sz
        block_mat_vec_mul_add!(C, A, B, sz)
    elseif n <  2sz && k >= 2sz && m >= 2sz
        block_covec_mat_mul_add!(C, A, B, sz)
    elseif n >= 2sz && k <  2sz && m >= 2sz
        block_vec_covec_mul_add!(C, A, B, sz)
    elseif n <  2sz && k >= 2sz && m <  2sz
        block_covec_vec_mul_add!(C, A, B, sz)
    else
        add_gemm_kernel!(C, A, B)
    end
end
