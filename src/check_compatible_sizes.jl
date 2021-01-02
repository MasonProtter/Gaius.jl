function check_compatible_sizes(C, A, B)
    n, m = size(C)
    a, k = size(A)
    b, c = size(B)
    @assert (n == a) && (m == c) && (k == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end

function check_compatible_sizes(C::VecTypes, A, B::VecTypes)
    n    = length(C)
    a, k = size(A)
    b    = length(B)
    @assert (n == a) && (k == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end

function check_compatible_sizes(C::CoVecTypes, A::CoVecTypes, B::MatTypes)
    m    = length(C)
    n    = length(A)
    a, b = size(B)
    @assert (n == a) && (m == b) "matrices of size $(size(C)), $(size(A)), $(size(B)) are incompatible"
    nothing
end
