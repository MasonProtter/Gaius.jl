function check_compatible_sizes(C, A, B)
    n, m = size(C)
    a, k = size(A)
    b, c = size(B)
    @assert (n == a) && (m == c) && (k == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end

function choose_block_size(C, A, B, ::Nothing)
    if (*)(length(C) |> Int128, length(A) |> Int128, length(B) |> Int128) >= ((3DEFAULT_BLOCK_SIZE) >>> 1)^6
        DEFAULT_BLOCK_SIZE
    else
        32
    end
end
choose_block_size(C, A, B, block_size::Integer) = block_size

function blocked_mul(A::MatTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Matrix{T}(undef, size(A,1), size(B,2))
    blocked_mul!(C, A, B)
    C
end

function blocked_mul(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Matrix{T}(undef, size(A, 1), size(B,2)),
                                 Matrix{T}(undef, size(A, 1), size(B,2))))
    blocked_mul!(C, A, B)
    C
end

function blocked_mul!(C::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T};
                      block_size = nothing, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C, A, B)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B _mul!(PtrArray(C), PtrArray(A), PtrArray(B), _block_size)
    C
end

function blocked_mul!(C::StructArray{Complex{T}}, A::StructArray{Complex{T}}, B::StructArray{Complex{T}};
                      block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.re, A.re, B.re)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B begin
        Cre, Cim = PtrArray(C.re), PtrArray(C.im)
        Are, Aim = PtrArray(A.re), PtrArray(A.im)
        Bre, Bim = PtrArray(B.re), PtrArray(B.im)
        _mul!(    Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    C
end

function blocked_mul!(C::Adjoint{Complex{T}, <:StructArray{Complex{T}}},
                      A::Adjoint{Complex{T}, <:StructArray{Complex{T}}},
                      B::StructArray{Complex{T}};
                      block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.parent.re', A.parent.re', B.re)

    _block_size = choose_block_size(C, A, B, block_size)
    A.parent.im .= (-).(A.parent.im) #ugly hack
    GC.@preserve C A B begin
        Cre, Cim = PtrArray(C.parent.re'), PtrArray(C.parent.im')
        Are, Aim = PtrArray(A.parent.re'), PtrArray(A.parent.im')
        Bre, Bim = PtrArray(B.re), PtrArray(B.im)
        _mul!(    Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    A.parent.im .= (-).(A.parent.im)
    C.parent.im .= (-).(C.parent.im) # ugly hack
    C
end

function blocked_mul!(C::Transpose{Complex{T}, <:StructArray{Complex{T}}},
                      A::Transpose{Complex{T}, <:StructArray{Complex{T}}},
                      B::StructArray{Complex{T}};
                      block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.parent.re |> transpose, A.parent.re |> transpose, B.re)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B begin
        Cre, Cim = PtrArray(C.parent.re |> transpose), PtrArray(C.parent.im |> transpose)
        Are, Aim = PtrArray(A.parent.re |> transpose), PtrArray(A.parent.im |> transpose)
        Bre, Bim = PtrArray(B.re), PtrArray(B.im)
        _mul!(    Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    C
end

function _mul!(C, A, B, sz)
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    if n >= sz+8 && m >= sz+8 && k >= sz+8
        block_mat_mat_mul!(C, A, B, sz)
    elseif n >= sz+8 && k >= sz+8 && m <  sz+8
        block_mat_vec_mul!(C, A, B, sz)
    elseif n <  sz+8 && k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(C, A, B, sz)
    elseif n >= sz+8 && k <  sz+8 && m >= sz+8
        block_vec_covec_mul!(C, A, B, sz)
    elseif n <  sz+8 && k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(C, A, B, sz)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    n, k, m = size(C, 1), size(A, 2), size(C, 2)
    if n >= sz+8 && m >= sz+8 && k >= sz+8
        block_mat_mat_mul_add!(C, A, B, sz, Val(factor))
    elseif n >= sz+8 && k >= sz+8 && m <  sz+8
        block_mat_vec_mul_add!(C, A, B, sz, Val(factor))
    elseif n <  sz+8 && k >= sz+8 && m >= sz+8
        block_covec_mat_mul_add!(C, A, B, sz, Val(factor))
    elseif n >= sz+8 && k <  sz+8 && m >= sz+8
        block_vec_covec_mul_add!(C, A, B, sz, Val(factor))
    elseif n <  sz+8 && k >= sz+8 && m <  sz+8
        block_covec_vec_mul_add!(C, A, B, sz, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end

#-----------------------------
# matvec

function check_compatible_sizes(C::VecTypes, A, B::VecTypes)
    n    = length(C)
    a, k = size(A)
    b    = length(B)
    @assert (n == a) && (k == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end

function blocked_mul(A::MatTypes, B::VecTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Vector{T}(undef, size(A,1))
    blocked_mul!(C, A, B)
    C
end

function blocked_mul(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 1}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(A, 1)),
                                 Vector{T}(undef, size(A, 1))))
    blocked_mul!(C, A, B)
    C
end

function _mul!(C::VecTypes{T}, A::MatTypes{T}, B::VecTypes{T}, sz) where {T<:Eltypes}
    n, k = size(A)
    if     n >= sz+8 && k >= sz+8
        block_mat_vec_mul!(C, A, B, sz)
    elseif n <  sz+8 && k >= sz+8
        block_covec_vec_mul!(C, A, B, sz)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(C::VecTypes{T}, A::MatTypes{T}, B::VecTypes{T}, sz, ::Val{factor} = Val(1)) where {factor, T<:Eltypes}
    n, k = size(A)
    if     n >= sz+8 && k >= sz+8
        block_mat_vec_mul_add!(C, A, B, sz, Val(factor))
    elseif n <  sz+8 && k >= sz+8
        block_covec_vec_mul_add!(C, A, B, sz, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end

#-----------------------------
# covec-mat

function check_compatible_sizes(C::CoVecTypes, A::CoVecTypes, B::MatTypes)
    m    = length(C)
    n    = length(A)
    a, b = size(B)
    @assert (n == a) && (m == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end

function blocked_mul(A::CoVecTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Vector{T}(undef, size(B,2))'
    blocked_mul!(C, A, B)
    C
end

function blocked_mul(A::Adjoint{Complex{T}, <:StructArray{Complex{T}, 1}}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(B, 2)),
                                 Vector{T}(undef, size(B, 2))))'
    blocked_mul!(C, A, B)
    C
end

function blocked_mul(A::Transpose{Complex{T}, <:StructArray{Complex{T}, 1}}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(B, 2)),
                                 Vector{T}(undef, size(B, 2)))) |> transpose
    blocked_mul!(C, A, B)
    C
end

function _mul!(C::CoVecTypes{T}, A::CoVecTypes{T}, B::MatTypes{T}, sz) where {T<:Eltypes}
    n, k, m = 1, size(A, 1), length(C)
    if     k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(C, A, B, sz)
    elseif k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(C, A, B, sz)
    else
        gemm_kernel!(C, A, B)
    end
end

function _mul_add!(C::CoVecTypes{T}, A::CoVecTypes{T}, B::MatTypes{T}, sz, ::Val{factor} = Val(1)) where {factor, T<:Eltypes}
    n, k, m = 1, size(A, 1), length(C)
    if     k >= sz+8 && m >= sz+8
        block_covec_mat_mul!(C, A, B, sz, Val(factor))
    elseif k >= sz+8 && m <  sz+8
        block_covec_vec_mul!(C, A, B, sz, Val(factor))
    else
        add_gemm_kernel!(C, A, B, Val(factor))
    end
end
