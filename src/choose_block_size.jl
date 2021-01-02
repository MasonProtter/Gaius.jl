function choose_block_size(C, A, B, ::Nothing)
    if (*)(length(C) |> Int128, length(A) |> Int128, length(B) |> Int128) >= ((3DEFAULT_BLOCK_SIZE) >>> 1)^6
        DEFAULT_BLOCK_SIZE
    else
        32
    end
end

choose_block_size(C, A, B, block_size::Integer) = block_size
