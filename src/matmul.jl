function _mul!(threading::Threading, C, A, B)
    sz = get_block_size(threading)
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    if n >= sz+8 && m >= sz+8 && k >= sz+8
        block_mat_mat_mul!(threading, C, A, B)
    elseif n >= sz+8 && k >= sz+8 && m <  sz+8
        block_mat_vec_mul!(threading, C, A, B)
    elseif n <  sz+8 && k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(threading, C, A, B)
    elseif n >= sz+8 && k <  sz+8 && m >= sz+8
        block_vec_covec_mul!(threading, C, A, B)
    elseif n <  sz+8 && k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(threading, C, A, B)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(threading::Threading, C, A, B, ::Val{factor} = Val(1)) where {factor}
    sz = get_block_size(threading)
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    if n >= sz+8 && m >= sz+8 && k >= sz+8
        block_mat_mat_mul_add!(threading, C, A, B, Val(factor))
    elseif n >= sz+8 && k >= sz+8 && m <  sz+8
        block_mat_vec_mul_add!(threading, C, A, B, Val(factor))
    elseif n <  sz+8 && k >= sz+8 && m >= sz+8
        block_covec_mat_mul_add!(threading, C, A, B, Val(factor))
    elseif n >= sz+8 && k <  sz+8 && m >= sz+8
        block_vec_covec_mul_add!(threading, C, A, B, Val(factor))
    elseif n <  sz+8 && k >= sz+8 && m <  sz+8
        block_covec_vec_mul_add!(threading, C, A, B, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end

function _mul!(threading::Threading, C::VecTypes{T}, A::MatTypes{T}, B::VecTypes{T}) where {T<:Eltypes}
    sz = get_block_size(threading)
    n, k = size(A)
    if     n >= sz+8 && k >= sz+8
        block_mat_vec_mul!(threading, C, A, B)
    elseif n <  sz+8 && k >= sz+8
        block_covec_vec_mul!(threading, C, A, B)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(threading::Threading, C::VecTypes{T}, A::MatTypes{T}, B::VecTypes{T}, ::Val{factor} = Val(1)) where {factor, T<:Eltypes}
    sz = get_block_size(threading)
    n, k = size(A)
    if     n >= sz+8 && k >= sz+8
        block_mat_vec_mul_add!(threading, C, A, B, Val(factor))
    elseif n <  sz+8 && k >= sz+8
        block_covec_vec_mul_add!(threading, C, A, B, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end

function _mul!(threading::Threading, C::CoVecTypes{T}, A::CoVecTypes{T}, B::MatTypes{T}) where {T<:Eltypes}
    sz = get_block_size(threading)
    n, k, m = 1, size(A, 1), length(C)
    if     k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(threading, C, A, B)
    elseif k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(threading, C, A, B)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(threading::Threading, C::CoVecTypes{T}, A::CoVecTypes{T}, B::MatTypes{T}, ::Val{factor} = Val(1)) where {factor, T<:Eltypes}
    sz = get_block_size(threading)
    n, k, m = 1, size(A, 1), length(C)
    if     k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(threading, C, A, B, Val(factor))
    elseif k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(threading, C, A, B, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end
