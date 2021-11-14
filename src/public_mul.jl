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
              block_size = nothing, singlethread_size = nothing,
              sizecheck = true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C, A, B)

    multithreaded = choose_multithread_parameter(C, A, B, singlethread_size, block_size)

    _mul!(multithreaded, C, A, B)
    C
end

function mul_serial!(C::AbstractArray{T}, A::AbstractArray{T}, B::AbstractArray{T};
                              block_size = nothing, sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C, A, B)

    singlethreaded = choose_singlethread_parameter(C, A, B, block_size)

    _mul!(singlethreaded, C, A, B)
    C
end

function mul!(C::StructArray{Complex{T}}, A::StructArray{Complex{T}}, B::StructArray{Complex{T}};
              block_size = default_block_size(), singlethread_size = nothing,
              sizecheck = true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.re, A.re, B.re)

    multithreaded = choose_multithread_parameter(C, A, B, singlethread_size, block_size)

    GC.@preserve C A B begin
        Cre, Cim = C.re, C.im
        Are, Aim = A.re, A.im
        Bre, Bim = B.re, B.im
        _mul!(    multithreaded,     Cre, Are, Bre)          # C.re = A.re * B.re
        _mul_add!(multithreaded,     Cre, Aim, Bim, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    multithreaded,     Cim, Are, Bim)          # C.im = A.re * B.im
        _mul_add!(multithreaded,     Cim, Aim, Bre)          # C.im = C.im + A.im * B.re
    end
    C
end

function mul_serial!(C::StructArray{Complex{T}}, A::StructArray{Complex{T}}, B::StructArray{Complex{T}};
                              block_size = default_block_size(), sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.re, A.re, B.re)

    singlethreaded = choose_singlethread_parameter(C, A, B, block_size)

    GC.@preserve C A B begin
        Cre, Cim = (C.re), (C.im)
        Are, Aim = (A.re), (A.im)
        Bre, Bim = (B.re), (B.im)
        _mul!(    singlethreaded,     Cre, Are, Bre)          # C.re = A.re * B.re
        _mul_add!(singlethreaded,     Cre, Aim, Bim, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    singlethreaded,     Cim, Are, Bim)          # C.im = A.re * B.im
        _mul_add!(singlethreaded,     Cim, Aim, Bre)          # C.im = C.im + A.im * B.re
    end
    C
end

function mul!(C::Adjoint{Complex{T}, <:StructArray{Complex{T}}},
              A::Adjoint{Complex{T}, <:StructArray{Complex{T}}},
              B::StructArray{Complex{T}};
              block_size = default_block_size(), singlethread_size = nothing,
              sizecheck = true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.parent.re', A.parent.re', B.re)

    multithreaded = choose_multithread_parameter(C, A, B, singlethread_size, block_size)
    A.parent.im .= (-).(A.parent.im) #ugly hack
    GC.@preserve C A B begin
        Cre, Cim = (C.parent.re'), (C.parent.im')
        Are, Aim = (A.parent.re'), (A.parent.im')
        Bre, Bim = (B.re), (B.im)
        _mul!(    multithreaded,     Cre, Are, Bre)          # C.re = A.re * B.re
        _mul_add!(multithreaded,     Cre, Aim, Bim, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    multithreaded,     Cim, Are, Bim)          # C.im = A.re * B.im
        _mul_add!(multithreaded,     Cim, Aim, Bre)          # C.im = C.im + A.im * B.re
    end
    A.parent.im .= (-).(A.parent.im)
    C.parent.im .= (-).(C.parent.im) # ugly hack
    C
end

function mul_serial!(C::Adjoint{Complex{T}, <:StructArray{Complex{T}}},
                              A::Adjoint{Complex{T}, <:StructArray{Complex{T}}},
                              B::StructArray{Complex{T}};
                              block_size = default_block_size(), sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.parent.re', A.parent.re', B.re)

    singlethreaded = choose_singlethread_parameter(C, A, B, block_size)
    A.parent.im .= (-).(A.parent.im) #ugly hack
    GC.@preserve C A B begin
        Cre, Cim = (C.parent.re'), (C.parent.im')
        Are, Aim = (A.parent.re'), (A.parent.im')
        Bre, Bim = (B.re), (B.im)
        _mul!(    singlethreaded,     Cre, Are, Bre)          # C.re = A.re * B.re
        _mul_add!(singlethreaded,     Cre, Aim, Bim, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    singlethreaded,     Cim, Are, Bim)          # C.im = A.re * B.im
        _mul_add!(singlethreaded,     Cim, Aim, Bre)          # C.im = C.im + A.im * B.re
    end
    A.parent.im .= (-).(A.parent.im)
    C.parent.im .= (-).(C.parent.im) # ugly hack
    C
end

function mul!(C::Transpose{Complex{T}, <:StructArray{Complex{T}}},
              A::Transpose{Complex{T}, <:StructArray{Complex{T}}},
              B::StructArray{Complex{T}};
              block_size = default_block_size(), singlethread_size = nothing,
              sizecheck = true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.parent.re |> transpose, A.parent.re |> transpose, B.re)

    multithreaded = choose_multithread_parameter(C, A, B, singlethread_size, block_size)

    GC.@preserve C A B begin
        Cre, Cim = (C.parent.re |> transpose), (C.parent.im |> transpose)
        Are, Aim = (A.parent.re |> transpose), (A.parent.im |> transpose)
        Bre, Bim = (B.re), (B.im)
        _mul!(    multithreaded,     Cre, Are, Bre)          # C.re = A.re * B.re
        _mul_add!(multithreaded,     Cre, Aim, Bim, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    multithreaded,     Cim, Are, Bim)          # C.im = A.re * B.im
        _mul_add!(multithreaded,     Cim, Aim, Bre)          # C.im = C.im + A.im * B.re
    end
    C
end

function mul_serial!(C::Transpose{Complex{T}, <:StructArray{Complex{T}}},
                              A::Transpose{Complex{T}, <:StructArray{Complex{T}}},
                              B::StructArray{Complex{T}};
                              block_size = default_block_size(), sizecheck=true) where {T <: Eltypes}
    sizecheck && check_compatible_sizes(C.parent.re |> transpose, A.parent.re |> transpose, B.re)

    singlethreaded = choose_singlethread_parameter(C, A, B, block_size)

    GC.@preserve C A B begin
        Cre, Cim = (C.parent.re |> transpose), (C.parent.im |> transpose)
        Are, Aim = (A.parent.re |> transpose), (A.parent.im |> transpose)
        Bre, Bim = (B.re), (B.im)
        _mul!(    singlethreaded,     Cre, Are, Bre)          # C.re = A.re * B.re
        _mul_add!(singlethreaded,     Cre, Aim, Bim, Val(-1)) # C.re = C.re - A.im * B.im
        _mul!(    singlethreaded,     Cim, Are, Bim)          # C.im = A.re * B.im
        _mul_add!(singlethreaded,     Cim, Aim, Bre)          # C.im = C.im + A.im * B.re
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
