#----------------------------------------------------------------
#----------------------------------------------------------------
# Block Matrix multiplication

@inline function block_mat_mat_mul!(C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @_sync begin
        @_spawn begin
            #_mul!(    C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        @_spawn begin
            _mul!(    C12, A11, B12, sz)
            _mul_add!(C12, A12, B22, sz)
        end
        @_spawn begin
            _mul!(    C21, A21, B11, sz)
            _mul_add!(C21, A22, B21, sz)
        end
        _mul!(    C22, A21, B12, sz)
        _mul_add!(C22, A22, B22, sz)
    end
end

function block_mat_vec_mul!(C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:end];
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    @_sync begin
        @_spawn begin
            #_mul!(    C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul!(    C21, A21, B11, sz)
        _mul_add!(C21, A22, B21, sz)
    end
end

function block_mat_vec_mul!(C::VecTypes, A, B::VecTypes, sz)
    @inbounds @views begin
        C11 = C[1:sz    ];
        C21 = C[sz+1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    @_sync begin
        @_spawn begin
            gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul!(    C21, A21, B11, sz)
        _mul_add!(C21, A22, B21, sz)
    end
end

function block_covec_mat_mul!(C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:end,    1:sz]; C12 = C[1:end,    sz+1:end]

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @_sync begin
        @_spawn begin
            #_mul!(    C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
            _mul_add!(C11, A12, B21, sz)
        end
        _mul!(    C12, A11, B12, sz)
        _mul_add!(C12, A12, B22, sz)
    end
end

function block_vec_covec_mul!(C, A, B, sz)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end];

        B11 = B[1:end,     1:sz]; B12 = B[1:end,     sz+1:end]
    end
    @_sync begin
        @_spawn begin
            #_mul!(    C11, A11, B11, sz)
            gemm_kernel!(C11, A11, B11)
        end
        @_spawn begin
            _mul!(C12, A11, B12, sz)
        end
        @_spawn begin
            _mul!(C21, A21, B11, sz)
        end
        _mul!(C22, A21, B12, sz)
    end
end

function block_covec_vec_mul!(C, A, B, sz)
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,     sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    gemm_kernel!(C, A11, B11)
    #_mul!(    C, A11, B11, sz)
    _mul_add!(C, A12, B21, sz)
end

function block_covec_vec_mul!(C::VecTypes, A, B::VecTypes, sz)
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,     sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    gemm_kernel!(C, A11, B11)
    _mul_add!(C, A12, B21, sz)
end

#----------------------------------------------------------------
#----------------------------------------------------------------
# Block Matrix addition-multiplication

@inline function block_mat_mat_mul_add!(C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
            _mul_add!(C11, A12, B21, sz, Val(factor))
        end
        @_spawn begin
            _mul_add!(C12, A11, B12, sz, Val(factor))
            _mul_add!(C12, A12, B22, sz, Val(factor))
        end
        @_spawn begin
            _mul_add!(C21, A21, B11, sz, Val(factor))
            _mul_add!(C21, A22, B21, sz, Val(factor))
        end
        _mul_add!(C22, A21, B12, sz, Val(factor))
        _mul_add!(C22, A22, B22, sz, Val(factor))
    end
end

function block_mat_vec_mul_add!(C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:end];
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
            _mul_add!(C11, A12, B21, sz, Val(factor))
        end
        _mul_add!(C21, A21, B11, sz, Val(factor))
        _mul_add!(C21, A22, B21, sz, Val(factor))
    end
end

function block_mat_vec_mul_add!(C::VecTypes, A, B::VecTypes, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz    ];
        C21 = C[sz+1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
            _mul_add!(C11, A12, B21, sz, Val(factor))
        end
        _mul_add!(C21, A21, B11, sz, Val(factor))
        _mul_add!(C21, A22, B21, sz, Val(factor))
    end
end

function block_covec_mat_mul_add!(C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:end,    1:sz]; C12 = C[1:end,     sz+1:end]

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
            _mul_add!(C11, A12, B21, sz, Val(factor))
        end
        _mul_add!(C12, A11, B12, sz, Val(factor))
        _mul_add!(C12, A12, B22, sz, Val(factor))
    end
end

function block_vec_covec_mul_add!(C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end];

        B11 = B[1:end,    1:sz]; B12 = B[1:end,    sz+1:end]
    end
    @_sync begin
        @_spawn begin
            add_gemm_kernel!(C11, A11, B11, Val(factor))
        end
        @_spawn begin
            _mul_add!(C12, A11, B12, sz, Val(factor))
        end
        @_spawn begin
            _mul_add!(C21, A21, B11, sz, Val(factor))
        end
        _mul_add!(C22, A21, B12, sz, Val(factor))
    end
end

function block_covec_vec_mul_add!(C, A, B, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    add_gemm_kernel!(C, A11, B11, Val(factor))
    _mul_add!(C, A12, B21, sz, Val(factor))
end

function block_covec_vec_mul_add!(C::VecTypes, A, B::VecTypes, sz, ::Val{factor} = Val(1)) where {factor}
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    add_gemm_kernel!(C, A11, B11, Val(factor))
    _mul_add!(C, A12, B21, sz, Val(factor))
end
