@inline function block_mat_mat_mul!(::Multithreaded, C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            #_mul!(multithreaded,     C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
            _mul_add!(multithreaded, C11, A12, B21, sz)
        end
        Threads.@spawn begin
            _mul!(multithreaded,     C12, A11, B12, sz)
            _mul_add!(multithreaded, C12, A12, B22, sz)
        end
        Threads.@spawn begin
            _mul!(multithreaded,     C21, A21, B11, sz)
            _mul_add!(multithreaded, C21, A22, B21, sz)
        end
        _mul!(multithreaded,     C22, A21, B12, sz)
        _mul_add!(multithreaded, C22, A22, B22, sz)
    end
end

@inline function block_mat_mat_mul!(::Singlethreaded, C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    #_mul!(singlethreaded,     C11, A11, B11, sz)
    gemm_kernel!(C11, A11, B11)
    _mul_add!(singlethreaded, C11, A12, B21, sz)
    _mul!(singlethreaded,     C12, A11, B12, sz)
    _mul_add!(singlethreaded, C12, A12, B22, sz)
    _mul!(singlethreaded,     C21, A21, B11, sz)
    _mul_add!(singlethreaded, C21, A22, B21, sz)
    _mul!(singlethreaded,     C22, A21, B12, sz)
    _mul_add!(singlethreaded, C22, A22, B22, sz)
end

function block_mat_vec_mul!(::Multithreaded, C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:end];
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    @sync begin
        Threads.@spawn begin
            #_mul!(multithreaded,     C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
            _mul_add!(multithreaded, C11, A12, B21, sz)
        end
        _mul!(multithreaded,     C21, A21, B11, sz)
        _mul_add!(multithreaded, C21, A22, B21, sz)
    end
end

function block_mat_vec_mul!(::Singlethreaded, C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:end];
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    #_mul!(singlethreaded,     C11, A11, B11, sz)
    gemm_kernel!(C11, A11, B11)
    _mul_add!(singlethreaded, C11, A12, B21, sz)
    _mul!(singlethreaded,     C21, A21, B11, sz)
    _mul_add!(singlethreaded, C21, A22, B21, sz)
end

function block_mat_vec_mul!(::Multithreaded, C::VecTypes, A, B::VecTypes, sz)
    @inbounds @views begin
        C11 = C[1:sz    ];
        C21 = C[sz+1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    @sync begin
        Threads.@spawn begin
            gemm_kernel!(C11, A11, B11)
            _mul_add!(multithreaded, C11, A12, B21, sz)
        end
        _mul!(multithreaded,     C21, A21, B11, sz)
        _mul_add!(multithreaded, C21, A22, B21, sz)
    end
end

function block_mat_vec_mul!(::Singlethreaded, C::VecTypes, A, B::VecTypes, sz)
    @inbounds @views begin
        C11 = C[1:sz    ];
        C21 = C[sz+1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    gemm_kernel!(C11, A11, B11)
    _mul_add!(singlethreaded, C11, A12, B21, sz)
    _mul!(singlethreaded,     C21, A21, B11, sz)
    _mul_add!(singlethreaded, C21, A22, B21, sz)
end

function block_covec_mat_mul!(::Multithreaded, C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:end,    1:sz]; C12 = C[1:end,    sz+1:end]

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            #_mul!(multithreaded,     C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
            _mul_add!(multithreaded, C11, A12, B21, sz)
        end
        _mul!(multithreaded,     C12, A11, B12, sz)
        _mul_add!(multithreaded, C12, A12, B22, sz)
    end
end

function block_covec_mat_mul!(::Singlethreaded, C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:end,    1:sz]; C12 = C[1:end,    sz+1:end]

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    #_mul!(threading,     C11, A11, B11, sz)
    gemm_kernel!(C11, A11, B11)
    _mul_add!(multithreaded, C11, A12, B21, sz)
    _mul!(multithreaded,     C12, A11, B12, sz)
    _mul_add!(multithreaded, C12, A12, B22, sz)
end

function block_vec_covec_mul!(::Multithreaded, C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end];

        B11 = B[1:end,     1:sz]; B12 = B[1:end,     sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            #_mul!(multithreaded,     C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
        end
        Threads.@spawn begin
            _mul!(multithreaded, C12, A11, B12, sz)
        end
        Threads.@spawn begin
            _mul!(multithreaded, C21, A21, B11, sz)
        end
        _mul!(multithreaded, C22, A21, B12, sz)
    end
end

function block_vec_covec_mul!(::Singlethreaded, C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end];

        B11 = B[1:end,     1:sz]; B12 = B[1:end,     sz+1:end]
    end
    #_mul!(singlethreaded,     C11, A11, B11, sz)
    gemm_kernel!(C11, A11, B11)
    _mul!(singlethreaded, C12, A11, B12, sz)
    _mul!(singlethreaded, C21, A21, B11, sz)
    _mul!(singlethreaded, C22, A21, B12, sz)
end

function block_covec_vec_mul!(threading::Threading, C, A, B, sz)
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,     sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    gemm_kernel!(C, A11, B11)
    #_mul!(threading,     C, A11, B11, sz)
    _mul_add!(threading, C, A12, B21, sz)
end

function block_covec_vec_mul!(threading::Threading, C::VecTypes, A, B::VecTypes, sz)
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,     sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    gemm_kernel!(C, A11, B11)
    _mul_add!(threading, C, A12, B21, sz)
end

@inline function block_mat_mat_mul_add!(::Multithreaded, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
            _mul_add!(multithreaded, C11, A12, B21, sz, Val(factor))
        end
        Threads.@spawn begin
            _mul_add!(multithreaded, C12, A11, B12, sz, Val(factor))
            _mul_add!(multithreaded, C12, A12, B22, sz, Val(factor))
        end
        Threads.@spawn begin
            _mul_add!(multithreaded, C21, A21, B11, sz, Val(factor))
            _mul_add!(multithreaded, C21, A22, B21, sz, Val(factor))
        end
        _mul_add!(multithreaded, C22, A21, B12, sz, Val(factor))
        _mul_add!(multithreaded, C22, A22, B22, sz, Val(factor))
    end
end

@inline function block_mat_mat_mul_add!(::Singlethreaded, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C11, A12, B21, sz, Val(factor))
    _mul_add!(singlethreaded, C12, A11, B12, sz, Val(factor))
    _mul_add!(singlethreaded, C12, A12, B22, sz, Val(factor))
    _mul_add!(singlethreaded, C21, A21, B11, sz, Val(factor))
    _mul_add!(singlethreaded, C21, A22, B21, sz, Val(factor))
    _mul_add!(singlethreaded, C22, A21, B12, sz, Val(factor))
    _mul_add!(singlethreaded, C22, A22, B22, sz, Val(factor))
end

function block_mat_vec_mul_add!(::Multithreaded, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:end];
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    @sync begin
        Threads.@spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
            _mul_add!(multithreaded, C11, A12, B21, sz, Val(factor))
        end
        _mul_add!(multithreaded, C21, A21, B11, sz, Val(factor))
        _mul_add!(multithreaded, C21, A22, B21, sz, Val(factor))
    end
end

function block_mat_vec_mul_add!(::Singlethreaded, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:end];
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C11, A12, B21, sz, Val(factor))
    _mul_add!(singlethreaded, C21, A21, B11, sz, Val(factor))
    _mul_add!(singlethreaded, C21, A22, B21, sz, Val(factor))
end

function block_mat_vec_mul_add!(::Multithreaded, C::VecTypes, A, B::VecTypes, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz    ];
        C21 = C[sz+1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    @sync begin
        Threads.@spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
            _mul_add!(multithreaded, C11, A12, B21, sz, Val(factor))
        end
        _mul_add!(multithreaded, C21, A21, B11, sz, Val(factor))
        _mul_add!(multithreaded, C21, A22, B21, sz, Val(factor))
    end
end

function block_mat_vec_mul_add!(::Singlethreaded, C::VecTypes, A, B::VecTypes, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz    ];
        C21 = C[sz+1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(multithreaded, C11, A12, B21, sz, Val(factor))
    _mul_add!(multithreaded, C21, A21, B11, sz, Val(factor))
    _mul_add!(multithreaded, C21, A22, B21, sz, Val(factor))
end

function block_covec_mat_mul_add!(::Multithreaded, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:end,    1:sz]; C12 = C[1:end,     sz+1:end]

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
            _mul_add!(multithreaded, C11, A12, B21, sz, Val(factor))
        end
        _mul_add!(multithreaded, C12, A11, B12, sz, Val(factor))
        _mul_add!(multithreaded, C12, A12, B22, sz, Val(factor))
    end
end

function block_covec_mat_mul_add!(::Singlethreaded, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:end,    1:sz]; C12 = C[1:end,     sz+1:end]

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C11, A12, B21, sz, Val(factor))
    _mul_add!(singlethreaded, C12, A11, B12, sz, Val(factor))
    _mul_add!(singlethreaded, C12, A12, B22, sz, Val(factor))
end

function block_vec_covec_mul_add!(::Multithreaded, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end];

        B11 = B[1:end,    1:sz]; B12 = B[1:end,    sz+1:end]
    end
    @sync begin
        Threads.@spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
        end
        Threads.@spawn begin
            _mul_add!(multithreaded, C12, A11, B12, sz, Val(factor))
        end
        Threads.@spawn begin
            _mul_add!(multithreaded, C21, A21, B11, sz, Val(factor))
        end
        _mul_add!(multithreaded, C22, A21, B12, sz, Val(factor))
    end
end

function block_vec_covec_mul_add!(::Singlethreaded, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end];

        B11 = B[1:end,    1:sz]; B12 = B[1:end,    sz+1:end]
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C12, A11, B12, sz, Val(factor))
    _mul_add!(singlethreaded, C21, A21, B11, sz, Val(factor))
    _mul_add!(singlethreaded, C22, A21, B12, sz, Val(factor))
end

function block_covec_vec_mul_add!(threading::Threading, C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    add_gemm_kernel!(C, A11, B11, Val(factor))
    _mul_add!(threading, C, A12, B21, sz, Val(factor))
end

function block_covec_vec_mul_add!(threading::Threading, C::VecTypes, A, B::VecTypes, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    add_gemm_kernel!(C, A11, B11, Val(factor))
    _mul_add!(threading, C, A12, B21, sz, Val(factor))
end
