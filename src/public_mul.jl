"""
    mul!(C, A, B)
    mul!(C, A, B; kwargs...)

Multiply A×B and store the result in C (overwriting the contents of C).

This function uses multithreading.

This function is part of the public API of Gaius.
"""
function mul! end

"""
    mul(A, B)
    mul(A, B; kwargs...)

Multiply A×B and return the result.

This function uses multithreading.

This function is part of the public API of Gaius.
"""
function mul end

"""
    mul_serial(A, B)
    mul_serial(A, B; kwargs...)

Multiply A×B and return the result.

This function will run single-threaded.

This function is part of the public API of Gaius.
"""
function mul_serial end

"""
    mul_serial!(C, A, B)
    mul_serial!(C, A, B; kwargs...)

Multiply A×B and store the result in C (overwriting the contents of C).

This function will run single-threaded.

This function is part of the public API of Gaius.
"""
function mul_serial! end

function mul(A::MatTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Matrix{T}(undef, size(A,1), size(B,2))
    mul!(C, A, B)
    C
end

function mul_serial(A::MatTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Matrix{T}(undef, size(A,1), size(B,2))
    mul_serial!(C, A, B)
    C
end

function mul(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Matrix{T}(undef, size(A, 1), size(B,2)),
                                 Matrix{T}(undef, size(A, 1), size(B,2))))
    mul!(C, A, B)
    C
end

function mul_serial(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Matrix{T}(undef, size(A, 1), size(B,2)),
                                 Matrix{T}(undef, size(A, 1), size(B,2))))
    mul_serial!(C, A, B)
    C
end

function mul!(C::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T};
              block_size = nothing, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C, A, B)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B _mul!(multithreaded, PtrArray(C), PtrArray(A), PtrArray(B), _block_size)
    C
end

function mul_serial!(C::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T};
                              block_size = nothing, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C, A, B)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B _mul!(singlethreaded, PtrArray(C), PtrArray(A), PtrArray(B), _block_size)
    C
end

function mul!(C::StructArray{Complex{T}}, A::StructArray{Complex{T}}, B::StructArray{Complex{T}};
              block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.re, A.re, B.re)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B begin
        Cre, Cim = PtrArray(C.re), PtrArray(C.im)
        Are, Aim = PtrArray(A.re), PtrArray(A.im)
        Bre, Bim = PtrArray(B.re), PtrArray(B.im)
        _mul!(    multithreaded,     Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(multithreaded,     Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    multithreaded,     Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(multithreaded,     Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    C
end

function mul_serial!(C::StructArray{Complex{T}}, A::StructArray{Complex{T}}, B::StructArray{Complex{T}};
                              block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.re, A.re, B.re)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B begin
        Cre, Cim = PtrArray(C.re), PtrArray(C.im)
        Are, Aim = PtrArray(A.re), PtrArray(A.im)
        Bre, Bim = PtrArray(B.re), PtrArray(B.im)
        _mul!(    singlethreaded,     Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(singlethreaded,     Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    singlethreaded,     Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(singlethreaded,     Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    C
end

function mul!(C::Adjoint{Complex{T}, <:StructArray{Complex{T}}},
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
        _mul!(    multithreaded,     Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(multithreaded,     Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    multithreaded,     Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(multithreaded,     Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    A.parent.im .= (-).(A.parent.im)
    C.parent.im .= (-).(C.parent.im) # ugly hack
    C
end

function mul_serial!(C::Adjoint{Complex{T}, <:StructArray{Complex{T}}},
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
        _mul!(    singlethreaded,     Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(singlethreaded,     Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    singlethreaded,     Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(singlethreaded,     Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    A.parent.im .= (-).(A.parent.im)
    C.parent.im .= (-).(C.parent.im) # ugly hack
    C
end

function mul!(C::Transpose{Complex{T}, <:StructArray{Complex{T}}},
              A::Transpose{Complex{T}, <:StructArray{Complex{T}}},
              B::StructArray{Complex{T}};
              block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.parent.re |> transpose, A.parent.re |> transpose, B.re)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B begin
        Cre, Cim = PtrArray(C.parent.re |> transpose), PtrArray(C.parent.im |> transpose)
        Are, Aim = PtrArray(A.parent.re |> transpose), PtrArray(A.parent.im |> transpose)
        Bre, Bim = PtrArray(B.re), PtrArray(B.im)
        _mul!(    multithreaded,     Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(multithreaded,     Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    multithreaded,     Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(multithreaded,     Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    C
end

function mul_serial!(C::Transpose{Complex{T}, <:StructArray{Complex{T}}},
                              A::Transpose{Complex{T}, <:StructArray{Complex{T}}},
                              B::StructArray{Complex{T}};
                              block_size = DEFAULT_BLOCK_SIZE, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.parent.re |> transpose, A.parent.re |> transpose, B.re)

    _block_size = choose_block_size(C, A, B, block_size)

    GC.@preserve C A B begin
        Cre, Cim = PtrArray(C.parent.re |> transpose), PtrArray(C.parent.im |> transpose)
        Are, Aim = PtrArray(A.parent.re |> transpose), PtrArray(A.parent.im |> transpose)
        Bre, Bim = PtrArray(B.re), PtrArray(B.im)
        _mul!(    singlethreaded,     Cre, Are, Bre, _block_size)          # C.re = A.re * B.re
        _mul_add!(singlethreaded,     Cre, Aim, Bim, _block_size, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    singlethreaded,     Cim, Are, Bim, _block_size)          # C.im = A.re * B.im
        _mul_add!(singlethreaded,     Cim, Aim, Bre, _block_size)          # C.im = C.im + A.im * B.re
    end
    C
end

function mul(A::MatTypes, B::VecTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Vector{T}(undef, size(A,1))
    mul!(C, A, B)
    C
end

function mul_serial(A::MatTypes, B::VecTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Vector{T}(undef, size(A,1))
    mul_serial!(C, A, B)
    C
end

function mul(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 1}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(A, 1)),
                                 Vector{T}(undef, size(A, 1))))
    mul!(C, A, B)
    C
end

function mul_serial(A::StructArray{Complex{T}, 2}, B::StructArray{Complex{T}, 1}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(A, 1)),
                                 Vector{T}(undef, size(A, 1))))
    mul_serial!(C, A, B)
    C
end

function mul(A::CoVecTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Vector{T}(undef, size(B,2))'
    mul!(C, A, B)
    C
end

function mul_serial(A::CoVecTypes, B::MatTypes)
    T = promote_type(eltype(A), eltype(B))
    C = Vector{T}(undef, size(B,2))'
    mul_serial!(C, A, B)
    C
end

function mul(A::Adjoint{Complex{T}, <:StructArray{Complex{T}, 1}},
             B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(B, 2)),
                                 Vector{T}(undef, size(B, 2))))'
    mul!(C, A, B)
    C
end

function mul_serial(A::Adjoint{Complex{T}, <:StructArray{Complex{T}, 1}},
                             B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(B, 2)),
                                 Vector{T}(undef, size(B, 2))))'
    mul_serial!(C, A, B)
    C
end

function mul(A::Transpose{Complex{T}, <:StructArray{Complex{T}, 1}},
             B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(B, 2)),
                                 Vector{T}(undef, size(B, 2)))) |> transpose
    mul!(C, A, B)
    C
end

function mul_serial(A::Transpose{Complex{T}, <:StructArray{Complex{T}, 1}},
                             B::StructArray{Complex{T}, 2}) where {T <: Eltypes}
    C = StructArray{Complex{T}}((Vector{T}(undef, size(B, 2)),
                                 Vector{T}(undef, size(B, 2)))) |> transpose
    mul_serial!(C, A, B)
    C
end
