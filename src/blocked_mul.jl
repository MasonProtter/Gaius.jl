"""
    blocked_mul(A::MatTypes, B::MatTypes)
"""
function blocked_mul(A::MatTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Matrix{T}(undef, size(A,1), size(B,2))
    blocked_mul!(C, A, B)
    C
end

"""
    blocked_mul(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
"""
function blocked_mul(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Matrix{T}(undef, size(A, 1), size(B,2)),
                                 Matrix{T}(undef, size(A, 1), size(B,2))))
    blocked_mul!(C, A, B)
    C
end

"""
    blocked_mul!(C::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T}; block_size = nothing, sizecheck=true) where {T <: Eltypes}
"""
function blocked_mul!(C::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T};
                      block_size = nothing, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C, A, B)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B _mul!(PtrArray(C), PtrArray(A), PtrArray(B), _block_size)
    C
end

"""
    blocked_mul!(C::StructArray{Complex{T}}, A::StructArray{Complex{T}}, B::StructArray{Complex{T}}; block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
"""
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

"""
    blocked_mul!(C::Adjoint{Complex{T}, <:StructArray{Complex{T}}}, A::Adjoint{Complex{T}, <:StructArray{Complex{T}}}, B::StructArray{Complex{T}}; block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
"""
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

"""
    blocked_mul!(C::Transpose{Complex{T}, <:StructArray{Complex{T}}}, A::Transpose{Complex{T}, <:StructArray{Complex{T}}}, B::StructArray{Complex{T}}; block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
"""
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

"""
    blocked_mul(A::MatTypes, B::VecTypes)
"""
function blocked_mul(A::MatTypes, B::VecTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Vector{T}(undef, size(A,1))
    blocked_mul!(C, A, B)
    C
end

"""
    blocked_mul(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 1}) where {T <: Eltypes}
"""
function blocked_mul(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 1}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(A, 1)),
                                 Vector{T}(undef, size(A, 1))))
    blocked_mul!(C, A, B)
    C
end

"""
    blocked_mul(A::CoVecTypes, B::MatTypes)
"""
function blocked_mul(A::CoVecTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Vector{T}(undef, size(B,2))'
    blocked_mul!(C, A, B)
    C
end

"""
    blocked_mul(A::Adjoint{Complex{T}, <:StructArray{Complex{T}, 1}}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
"""
function blocked_mul(A::Adjoint{Complex{T}, <:StructArray{Complex{T}, 1}},
                     B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(B, 2)),
                                 Vector{T}(undef, size(B, 2))))'
    blocked_mul!(C, A, B)
    C
end

"""
    blocked_mul(A::Transpose{Complex{T}, <:StructArray{Complex{T}, 1}}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
"""
function blocked_mul(A::Transpose{Complex{T}, <:StructArray{Complex{T}, 1}},
                     B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(B, 2)),
                                 Vector{T}(undef, size(B, 2)))) |> transpose
    blocked_mul!(C, A, B)
    C
end
