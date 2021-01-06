function _mul!(threading::Threading, C, A, B, sz)
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    if n >= sz+8 && m >= sz+8 && k >= sz+8
        block_mat_mat_mul!(threading, C, A, B, sz)
    elseif n >= sz+8 && k >= sz+8 && m <  sz+8
        block_mat_vec_mul!(threading, C, A, B, sz)
    elseif n <  sz+8 && k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(threading, C, A, B, sz)
    elseif n >= sz+8 && k <  sz+8 && m >= sz+8
        block_vec_covec_mul!(threading, C, A, B, sz)
    elseif n <  sz+8 && k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(threading, C, A, B, sz)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(threading::Threading, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    if n >= sz+8 && m >= sz+8 && k >= sz+8
        block_mat_mat_mul_add!(threading, C, A, B, sz, Val(factor))
    elseif n >= sz+8 && k >= sz+8 && m <  sz+8
        block_mat_vec_mul_add!(threading, C, A, B, sz, Val(factor))
    elseif n <  sz+8 && k >= sz+8 && m >= sz+8
        block_covec_mat_mul_add!(threading, C, A, B, sz, Val(factor))
    elseif n >= sz+8 && k <  sz+8 && m >= sz+8
        block_vec_covec_mul_add!(threading, C, A, B, sz, Val(factor))
    elseif n <  sz+8 && k >= sz+8 && m <  sz+8
        block_covec_vec_mul_add!(threading, C, A, B, sz, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end

function _mul!(threading::Threading, C::VecTypes{T}, A::MatTypes{T}, B::VecTypes{T}, sz) where {T<:Eltypes}
    n, k = size(A)
    if     n >= sz+8 && k >= sz+8
        block_mat_vec_mul!(threading, C, A, B, sz)
    elseif n <  sz+8 && k >= sz+8
        block_covec_vec_mul!(threading, C, A, B, sz)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(threading::Threading, C::VecTypes{T}, A::MatTypes{T}, B::VecTypes{T}, sz, ::Val{factor} = Val(1)) where {factor, T<:Eltypes}
    n, k = size(A)
    if     n >= sz+8 && k >= sz+8
        block_mat_vec_mul_add!(threading, C, A, B, sz, Val(factor))
    elseif n <  sz+8 && k >= sz+8
        block_covec_vec_mul_add!(threading, C, A, B, sz, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end

function _mul!(threading::Threading, C::CoVecTypes{T}, A::CoVecTypes{T}, B::MatTypes{T}, sz) where {T<:Eltypes}
    n, k, m = 1, size(A, 1), length(C)
    if     k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(threading, C, A, B, sz)
    elseif k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(threading, C, A, B, sz)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(threading::Threading, C::CoVecTypes{T}, A::CoVecTypes{T}, B::MatTypes{T}, sz, ::Val{factor} = Val(1)) where {factor, T<:Eltypes}
    n, k, m = 1, size(A, 1), length(C)
    if     k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(threading, C, A, B, sz, Val(factor))
    elseif k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(threading, C, A, B, sz, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end
