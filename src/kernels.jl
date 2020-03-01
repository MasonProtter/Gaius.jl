function gemm_kernel!(C, A, B)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += A[n,k] * B[k,m]
        end
        C[n,m] = Cₙₘ
    end
end

function add_gemm_kernel!(C, A, B, ::Val{factor}) where {factor}
    if factor == 1
        _add_gemm_kernel!(C, A, B)
    elseif factor == -1
        _sub_gemm_kernel!(C, A, B)
    else
        _add_gemm_kernel!(C, C, B, Val(factor))
    end
end

function _add_gemm_kernel!(C, A, B)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ += A[n,k] * B[k,m]
        end
        C[n,m] += Cₙₘ
    end
end

function _sub_gemm_kernel!(C, A, B)
    @avx for n ∈ 1:size(A, 1), m ∈ 1:size(B, 2)
        Cₙₘ = zero(eltype(C))
        for k ∈ 1:size(A, 2)
            Cₙₘ -= A[n,k] * B[k,m]
        end
        C[n,m] += Cₙₘ
    end
end

_add_gemm_kernel!(C, A, B, ::Val{1}) = _add_gemm_kernel!(C, A, B)
_add_gemm_kernel!(C, A, B, ::Val{-1}) = _sub_gemm_kernel!(C, A, B)

