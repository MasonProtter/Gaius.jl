@inline function block_mat_mat_mul!(multithreaded::Multithreaded, C, A, B)
    use_singlethread(multithreaded, C, A, B) &&
        return block_mat_mat_mul!(multithreaded.singlethreaded, C, A, B)
    sz = get_block_size(multithreaded)
    mᵣ, nᵣ = LoopVectorization.matmul_params()
    mstep = mᵣ * LoopVectorization.pick_vector_width(eltype(A))
    m1 = min(max(sz, mstep * div(size(A, 1), 2mstep, RoundNearest)), size(A, 1) - sz)
    n1 = min(max(sz, nᵣ * div(size(B, 2), 2nᵣ, RoundNearest)), size(B, 2) - sz)
    k1 = min(max(sz, div(size(A, 2), 2, RoundNearest)), size(A, 2) - sz)
    @inbounds @views begin
        C11 = C[1:m1,     1:n1]; C12 = C[1:m1,     n1+1:end]
        C21 = C[m1+1:end, 1:n1]; C22 = C[m1+1:end, n1+1:end]

        A11 = A[1:m1,     1:k1]; A12 = A[1:m1,     k1+1:end]
        A21 = A[m1+1:end, 1:k1]; A22 = A[m1+1:end, k1+1:end]

        B11 = B[1:k1,     1:n1]; B12 = B[1:k1,     n1+1:end]
        B21 = B[k1+1:end, 1:n1]; B22 = B[k1+1:end, n1+1:end]
    end
    @sync begin
        Threads.@spawn begin
            _mul!(multithreaded,     C11, A11, B11)
            _mul_add!(multithreaded, C11, A12, B21)
        end
        Threads.@spawn begin
            _mul!(multithreaded,     C12, A11, B12)
            _mul_add!(multithreaded, C12, A12, B22)
        end
        Threads.@spawn begin
            _mul!(multithreaded,     C21, A21, B11)
            _mul_add!(multithreaded, C21, A22, B21)
        end
        _mul!(multithreaded,     C22, A21, B12)
        _mul_add!(multithreaded, C22, A22, B22)
    end
end

@inline function block_mat_mat_mul!(singlethreaded::Singlethreaded, C, A, B)
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    #_mul!(singlethreaded,     C11, A11, B11)
    gemm_kernel!(C11, A11, B11)
    _mul_add!(singlethreaded, C11, A12, B21)
    _mul!(singlethreaded,     C12, A11, B12)
    _mul_add!(singlethreaded, C12, A12, B22)
    _mul!(singlethreaded,     C21, A21, B11)
    _mul_add!(singlethreaded, C21, A22, B21)
    _mul!(singlethreaded,     C22, A21, B12)
    _mul_add!(singlethreaded, C22, A22, B22)
end

function block_mat_vec_mul!(multithreaded::Multithreaded, C, A, B)
    use_singlethread(multithreaded, C, A, B) &&
        return block_mat_vec_mul!(multithreaded.singlethreaded, C, A, B)
    sz = get_block_size(multithreaded)
    mstep = LoopVectorization.pick_vector_width(eltype(A))
    m1 = min(max(sz, mstep * div(size(A, 1), 2mstep, RoundNearest)), size(A, 1) - sz)
    k1 = min(max(sz, div(size(A, 2), 2, RoundNearest)), size(A, 2) - sz)
    @inbounds @views begin
        C11 = C[1:m1,     1:end];
        C21 = C[m1+1:end, 1:end];

        A11 = A[1:m1,     1:k1]; A12 = A[1:m1,     k1+1:end]
        A21 = A[m1+1:end, 1:k1]; A22 = A[m1+1:end, k1+1:end]

        B11 = B[1:k1,     1:end];
        B21 = B[k1+1:end, 1:end];
    end
    @sync begin
        Threads.@spawn begin
            _mul!(multithreaded,     C11, A11, B11)
            _mul_add!(multithreaded, C11, A12, B21)
        end
        _mul!(multithreaded,     C21, A21, B11)
        _mul_add!(multithreaded, C21, A22, B21)
    end
end

function block_mat_vec_mul!(singlethreaded::Singlethreaded, C, A, B)
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:sz,     1:end];
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    #_mul!(singlethreaded,     C11, A11, B11)
    gemm_kernel!(C11, A11, B11)
    _mul_add!(singlethreaded, C11, A12, B21)
    _mul!(singlethreaded,     C21, A21, B11)
    _mul_add!(singlethreaded, C21, A22, B21)
end

function block_mat_vec_mul!(multithreaded::Multithreaded, C::VecTypes, A, B::VecTypes)
    use_singlethread(multithreaded, C, A, B) &&
        return block_mat_vec_mul!(multithreaded.singlethreaded, C, A, B)
    sz = get_block_size(multithreaded)
    mstep = LoopVectorization.pick_vector_width(eltype(A))
    m1 = min(max(sz, mstep * div(size(A, 1), 2mstep, RoundNearest)), size(A, 1) - sz)
    k1 = min(max(sz, div(size(A, 2), 2, RoundNearest)), size(A, 2) - sz)
    @inbounds @views begin
        C11 = C[1:m1    ];
        C21 = C[m1+1:end];

        A11 = A[1:m1,     1:k1]; A12 = A[1:m1,     k1+1:end]
        A21 = A[m1+1:end, 1:k1]; A22 = A[m1+1:end, k1+1:end]

        B11 = B[1:k1    ];
        B21 = B[k1+1:end];
    end
    @sync begin
        Threads.@spawn begin
            _mul!(multithreaded,     C11, A11, B11)
            _mul_add!(multithreaded, C11, A12, B21)
        end
        _mul!(multithreaded,     C21, A21, B11)
        _mul_add!(multithreaded, C21, A22, B21)
    end
end

function block_mat_vec_mul!(singlethreaded::Singlethreaded, C::VecTypes, A, B::VecTypes)
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:sz    ];
        C21 = C[sz+1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    gemm_kernel!(C11, A11, B11)
    _mul_add!(singlethreaded, C11, A12, B21)
    _mul!(singlethreaded,     C21, A21, B11)
    _mul_add!(singlethreaded, C21, A22, B21)
end

function block_covec_mat_mul!(multithreaded::Multithreaded, C, A, B)
    use_singlethread(multithreaded, C, A, B) &&
        return block_covec_mat_mul!(multithreaded.singlethreaded, C, A, B)
    sz = get_block_size(multithreaded)
    nstep = LoopVectorization.pick_vector_width(eltype(B))
    n1 = min(max(sz, nstep * div(size(B, 2), 2nstep, RoundNearest)), size(B, 2) - sz)
    k1 = min(max(sz, div(size(A, 2), 2, RoundNearest)), size(A, 2) - sz)
    @inbounds @views begin
        C11 = C[1:end,    1:n1]; C12 = C[1:end,    n1+1:end]

        A11 = A[1:end,    1:k1]; A12 = A[1:end,    k1+1:end]

        B11 = B[1:k1,     1:n1]; B12 = B[1:k1,     n1+1:end]
        B21 = B[k1+1:end, 1:n1]; B22 = B[k1+1:end, n1+1:end]
    end
    @sync begin
        Threads.@spawn begin
            _mul!(multithreaded,     C11, A11, B11)
            _mul_add!(multithreaded, C11, A12, B21)
        end
        _mul!(multithreaded,     C12, A11, B12)
        _mul_add!(multithreaded, C12, A12, B22)
    end
end

function block_covec_mat_mul!(singlethreaded::Singlethreaded, C, A, B)
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:end,    1:sz]; C12 = C[1:end,    sz+1:end]

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    #_mul!(threading,     C11, A11, B11)
    gemm_kernel!(C11, A11, B11)
    _mul_add!(singlethreaded, C11, A12, B21)
    _mul!(singlethreaded,     C12, A11, B12)
    _mul_add!(singlethreaded, C12, A12, B22)
end

function block_vec_covec_mul!(multithreaded::Multithreaded, C, A, B)
    use_singlethread(multithreaded, C, A, B) &&
        return block_vec_covec_mul!(multithreaded.singlethreaded, C, A, B)
    sz = get_block_size(multithreaded)
    mᵣ, nᵣ = LoopVectorization.matmul_params()
    mstep = mᵣ * LoopVectorization.pick_vector_width(eltype(A))
    m1 = min(max(sz, mstep * div(size(A, 1), 2mstep, RoundNearest)), size(A, 1) - sz)
    n1 = min(max(sz, nᵣ * div(size(B, 2), 2nᵣ, RoundNearest)), size(B, 2) - sz)
    @inbounds @views begin
        C11 = C[1:m1,     1:n1]; C12 = C[1:m1,     n1+1:end]
        C21 = C[m1+1:end, 1:n1]; C22 = C[m1+1:end, n1+1:end]

        A11 = A[1:m1,     1:end];
        A21 = A[m1+1:end, 1:end];

        B11 = B[1:end,     1:n1]; B12 = B[1:end,     n1+1:end]
    end
    @sync begin
        Threads.@spawn begin
            _mul!(multithreaded, C11, A11, B11)
        end
        Threads.@spawn begin
            _mul!(multithreaded, C12, A11, B12)
        end
        Threads.@spawn begin
            _mul!(multithreaded, C21, A21, B11)
        end
        _mul!(multithreaded, C22, A21, B12)
    end
end

function block_vec_covec_mul!(singlethreaded::Singlethreaded, C, A, B)
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end];

        B11 = B[1:end,     1:sz]; B12 = B[1:end,     sz+1:end]
    end
    #_mul!(singlethreaded,     C11, A11, B11)
    gemm_kernel!(C11, A11, B11)
    _mul!(singlethreaded, C12, A11, B12)
    _mul!(singlethreaded, C21, A21, B11)
    _mul!(singlethreaded, C22, A21, B12)
end

function block_covec_vec_mul!(threading::Threading, C, A, B)
    sz = get_block_size(threading)
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,     sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    gemm_kernel!(C, A11, B11)
    #_mul!(threading,     C, A11, B11)
    _mul_add!(threading, C, A12, B21)
end

function block_covec_vec_mul!(threading::Threading, C::VecTypes, A, B::VecTypes)
    sz = get_block_size(threading)
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,     sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    gemm_kernel!(C, A11, B11)
    _mul_add!(threading, C, A12, B21)
end

@inline function block_mat_mat_mul_add!(multithreaded::Multithreaded, C, A, B, ::Val{factor} = Val(1)) where {factor}
    use_singlethread(multithreaded, C, A, B) &&
        return block_mat_mat_mul_add!(multithreaded.singlethreaded, C, A, B, Val(factor))
    sz = get_block_size(multithreaded)
    mᵣ, nᵣ = LoopVectorization.matmul_params()
    mstep = mᵣ * LoopVectorization.pick_vector_width(eltype(A))
    m1 = min(max(sz, mstep * div(size(A, 1), 2mstep, RoundNearest)), size(A, 1) - sz)
    n1 = min(max(sz, nᵣ * div(size(B, 2), 2nᵣ, RoundNearest)), size(B, 2) - sz)
    k1 = min(max(sz, div(size(A, 2), 2, RoundNearest)), size(A, 2) - sz)
    @inbounds @views begin
        C11 = C[1:m1,     1:n1]; C12 = C[1:m1,     n1+1:end]
        C21 = C[m1+1:end, 1:n1]; C22 = C[m1+1:end, n1+1:end]

        A11 = A[1:m1,     1:k1]; A12 = A[1:m1,     k1+1:end]
        A21 = A[m1+1:end, 1:k1]; A22 = A[m1+1:end, k1+1:end]

        B11 = B[1:k1,     1:n1]; B12 = B[1:k1,     n1+1:end]
        B21 = B[k1+1:end, 1:n1]; B22 = B[k1+1:end, n1+1:end]
    end
    @sync begin
        Threads.@spawn begin
            _mul_add!(multithreaded, C11, A11, B11, Val(factor))
            _mul_add!(multithreaded, C11, A12, B21, Val(factor))
        end
        Threads.@spawn begin
            _mul_add!(multithreaded, C12, A11, B12, Val(factor))
            _mul_add!(multithreaded, C12, A12, B22, Val(factor))
        end
        Threads.@spawn begin
            _mul_add!(multithreaded, C21, A21, B11, Val(factor))
            _mul_add!(multithreaded, C21, A22, B21, Val(factor))
        end
        _mul_add!(multithreaded, C22, A21, B12, Val(factor))
        _mul_add!(multithreaded, C22, A22, B22, Val(factor))
    end
end

@inline function block_mat_mat_mul_add!(singlethreaded::Singlethreaded, C, A, B, ::Val{factor} = Val(1)) where {factor}
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C11, A12, B21, Val(factor))
    _mul_add!(singlethreaded, C12, A11, B12, Val(factor))
    _mul_add!(singlethreaded, C12, A12, B22, Val(factor))
    _mul_add!(singlethreaded, C21, A21, B11, Val(factor))
    _mul_add!(singlethreaded, C21, A22, B21, Val(factor))
    _mul_add!(singlethreaded, C22, A21, B12, Val(factor))
    _mul_add!(singlethreaded, C22, A22, B22, Val(factor))
end

function block_mat_vec_mul_add!(multithreaded::Multithreaded, C, A, B, ::Val{factor} = Val(1)) where {factor}
    use_singlethread(multithreaded, C, A, B) &&
        return block_mat_vec_mul_add!(multithreaded.singlethreaded, C, A, B, Val(factor))
    sz = get_block_size(multithreaded)
    mstep = LoopVectorization.pick_vector_width(eltype(A))
    m1 = min(max(sz, mstep * div(size(A, 1), 2mstep, RoundNearest)), size(A, 1) - sz)
    k1 = min(max(sz, div(size(A, 2), 2, RoundNearest)), size(A, 2) - sz)
    @inbounds @views begin
        C11 = C[1:m1,     1:end];
        C21 = C[m1+1:end, 1:end];

        A11 = A[1:m1,     1:k1]; A12 = A[1:m1,     k1+1:end]
        A21 = A[m1+1:end, 1:k1]; A22 = A[m1+1:end, k1+1:end]

        B11 = B[1:k1,     1:end];
        B21 = B[k1+1:end, 1:end];
    end
    @sync begin
        Threads.@spawn begin
            _mul_add!(multithreaded, C11, A11, B11, Val(factor))
            _mul_add!(multithreaded, C11, A12, B21, Val(factor))
        end
        _mul_add!(multithreaded, C21, A21, B11, Val(factor))
        _mul_add!(multithreaded, C21, A22, B21, Val(factor))
    end
end

function block_mat_vec_mul_add!(singlethreaded::Singlethreaded, C, A, B, ::Val{factor} = Val(1)) where {factor}
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:sz,     1:end];
        C21 = C[sz+1:end, 1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C11, A12, B21, Val(factor))
    _mul_add!(singlethreaded, C21, A21, B11, Val(factor))
    _mul_add!(singlethreaded, C21, A22, B21, Val(factor))
end

function block_mat_vec_mul_add!(multithreaded::Multithreaded, C::VecTypes, A, B::VecTypes, ::Val{factor} = Val(1)) where {factor}
    use_singlethread(multithreaded, C, A, B) &&
        return block_mat_vec_mul_add!(multithreaded.singlethreaded, C, A, B, Val(factor))
    sz = get_block_size(multithreaded)
    mstep = LoopVectorization.pick_vector_width(eltype(A))
    m1 = min(max(sz, mstep * div(size(A, 1), 2mstep, RoundNearest)), size(A, 1) - sz)
    k1 = min(max(sz, div(size(A, 2), 2, RoundNearest)), size(A, 2) - sz)
    @inbounds @views begin
        C11 = C[1:m1,   ];
        C21 = C[m1+1:end];

        A11 = A[1:m1,     1:k1]; A12 = A[1:m1,     k1+1:end]
        A21 = A[m1+1:end, 1:k1]; A22 = A[m1+1:end, k1+1:end]

        B11 = B[1:k1,   ];
        B21 = B[k1+1:end];
    end
    @sync begin
        Threads.@spawn begin
            _mul_add!(multithreaded, C11, A11, B11, Val(factor))
            _mul_add!(multithreaded, C11, A12, B21, Val(factor))
        end
        _mul_add!(multithreaded, C21, A21, B11, Val(factor))
        _mul_add!(multithreaded, C21, A22, B21, Val(factor))
    end
end

function block_mat_vec_mul_add!(singlethreaded::Singlethreaded, C::VecTypes, A, B::VecTypes, ::Val{factor} = Val(1)) where {factor}
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:sz    ];
        C21 = C[sz+1:end];

        A11 = A[1:sz,     1:sz]; A12 = A[1:sz,     sz+1:end]
        A21 = A[sz+1:end, 1:sz]; A22 = A[sz+1:end, sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C11, A12, B21, Val(factor))
    _mul_add!(singlethreaded, C21, A21, B11, Val(factor))
    _mul_add!(singlethreaded, C21, A22, B21, Val(factor))
end

function block_covec_mat_mul_add!(multithreaded::Multithreaded, C, A, B, ::Val{factor} = Val(1)) where {factor}
    use_singlethread(multithreaded, C, A, B) &&
        return block_covec_mat_mul_add!(multithreaded.singlethreaded, C, A, B, Val(factor))
    sz = get_block_size(multithreaded)
    nstep = LoopVectorization.pick_vector_width(eltype(B))
    n1 = min(max(sz, nstep * div(size(B, 2), 2nstep, RoundNearest)), size(B, 2) - sz)
    k1 = min(max(sz, div(size(A, 2), 2, RoundNearest)), size(A, 2) - sz)
    @inbounds @views begin
        C11 = C[1:end,    1:n1]; C12 = C[1:end,    n1+1:end]

        A11 = A[1:end,    1:k1]; A12 = A[1:end,    k1+1:end]

        B11 = B[1:k1,     1:n1]; B12 = B[1:k1,     n1+1:end]
        B21 = B[k1+1:end, 1:n1]; B22 = B[k1+1:end, n1+1:end]
    end
    @sync begin
        Threads.@spawn begin
            _mul_add!(multithreaded, C11, A11, B11, Val(factor))
            _mul_add!(multithreaded, C11, A12, B21, Val(factor))
        end
        _mul_add!(multithreaded, C12, A11, B12, Val(factor))
        _mul_add!(multithreaded, C12, A12, B22, Val(factor))
    end
end

function block_covec_mat_mul_add!(singlethreaded::Singlethreaded, C, A, B, ::Val{factor} = Val(1)) where {factor}
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:end,    1:sz]; C12 = C[1:end,     sz+1:end]

        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:sz]; B12 = B[1:sz,     sz+1:end]
        B21 = B[sz+1:end, 1:sz]; B22 = B[sz+1:end, sz+1:end]
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C11, A12, B21, Val(factor))
    _mul_add!(singlethreaded, C12, A11, B12, Val(factor))
    _mul_add!(singlethreaded, C12, A12, B22, Val(factor))
end

function block_vec_covec_mul_add!(multithreaded::Multithreaded, C, A, B, ::Val{factor} = Val(1)) where {factor}
    use_singlethread(multithreaded, C, A, B) &&
        return block_vec_covec_mul_add!(multithreaded.singlethreaded, C, A, B, Val(factor))
    sz = get_block_size(multithreaded)
    mᵣ, nᵣ = LoopVectorization.matmul_params()
    mstep = mᵣ * LoopVectorization.pick_vector_width(eltype(A))
    m1 = min(max(sz, mstep * div(size(A, 1), 2mstep, RoundNearest)), size(A, 1) - sz)
    n1 = min(max(sz, nᵣ * div(size(B, 2), 2nᵣ, RoundNearest)), size(B, 2) - sz)
    @inbounds @views begin
        C11 = C[1:m1,     1:n1]; C12 = C[1:m1,     n1+1:end]
        C21 = C[m1+1:end, 1:n1]; C22 = C[m1+1:end, n1+1:end]

        A11 = A[1:m1,     1:end];
        A21 = A[m1+1:end, 1:end];

        B11 = B[1:end,    1:n1]; B12 = B[1:end,    n1+1:end]
    end
    @sync begin
        Threads.@spawn begin
            _mul_add!(multithreaded, C11, A11, B11, Val(factor))
        end
        Threads.@spawn begin
            _mul_add!(multithreaded, C12, A11, B12, Val(factor))
        end
        Threads.@spawn begin
            _mul_add!(multithreaded, C21, A21, B11, Val(factor))
        end
        _mul_add!(multithreaded, C22, A21, B12, Val(factor))
    end
end

function block_vec_covec_mul_add!(singlethreaded::Singlethreaded, C, A, B, ::Val{factor} = Val(1)) where {factor}
    sz = get_block_size(singlethreaded)
    @inbounds @views begin
        C11 = C[1:sz,     1:sz]; C12 = C[1:sz,     sz+1:end]
        C21 = C[sz+1:end, 1:sz]; C22 = C[sz+1:end, sz+1:end]

        A11 = A[1:sz,     1:end];
        A21 = A[sz+1:end, 1:end];

        B11 = B[1:end,    1:sz]; B12 = B[1:end,    sz+1:end]
    end
    add_gemm_kernel!(C11, A11, B11, Val(factor))
    _mul_add!(singlethreaded, C12, A11, B12, Val(factor))
    _mul_add!(singlethreaded, C21, A21, B11, Val(factor))
    _mul_add!(singlethreaded, C22, A21, B12, Val(factor))
end

function block_covec_vec_mul_add!(threading::Threading, C, A, B, ::Val{factor} = Val(1)) where {factor}
    sz = get_block_size(threading)
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz,     1:end];
        B21 = B[sz+1:end, 1:end];
    end
    add_gemm_kernel!(C, A11, B11, Val(factor))
    _mul_add!(threading, C, A12, B21, Val(factor))
end

function block_covec_vec_mul_add!(threading::Threading, C::VecTypes, A, B::VecTypes, ::Val{factor} = Val(1)) where {factor}
    sz = get_block_size(threading)
    @inbounds @views begin
        A11 = A[1:end,    1:sz]; A12 = A[1:end,    sz+1:end]

        B11 = B[1:sz    ];
        B21 = B[sz+1:end];
    end
    add_gemm_kernel!(C, A11, B11, Val(factor))
    _mul_add!(threading, C, A12, B21, Val(factor))
end
