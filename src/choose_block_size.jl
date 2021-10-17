function choose_block_size(C, A, B, ::Nothing)
    if (*)(length(C) |> Int128, length(A) |> Int128, length(B) |> Int128) >= ((3default_block_size()) >>> 1)^6
        default_block_size()
    else
        32
    end
end

choose_block_size(C, A, B, block_size::Integer) = block_size

function choose_parameter(C, A, B, ::Nothing)
    m = Int128(size(A, 1))
    n = Int128(size(B, 1))
    k = Int128(size(A, 2))
    oversubscription_ratio = 8  # a magic constant that Works For Me
    singlethread_size = cld(m * n * k, oversubscription_ratio * Threads.nthreads())
    return Multithreaded(singlethread_size)
end

choose_parameter(C, A, B, singlethread_size::Integer) = Multithreaded(singlethread_size)

function use_singlethread(multithreaded::Multithreaded, C, A, B)
    m = Int128(size(A, 1))
    n = Int128(size(B, 1))
    k = Int128(size(A, 2))
    return m * n * k <= multithreaded.singlethread_size
end
