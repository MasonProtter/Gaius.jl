function choose_block_size(C, A, B, ::Nothing)
    if (*)(length(C) |> Int128, length(A) |> Int128, length(B) |> Int128) >= ((3default_block_size()) >>> 1)^6
        default_block_size()
    else
        32
    end
end

choose_block_size(C, A, B, block_size::Integer) = block_size
