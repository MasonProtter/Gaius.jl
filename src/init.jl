function __init__()
    _print_num_threads_warning()
    return nothing
end

function _print_num_threads_warning()
    sys_nc = Int(VectorizationBase.num_cores())::Int
    jl_nt = Threads.nthreads()
    return _print_num_threads_warning(Int(sys_nc), jl_nt)
end

function _print_num_threads_warning(sys_nc::Int, jl_nt::Int)
    if jl_nt < sys_nc
        if !_is_suppress_warning()
            msg = string(
                "The system has $(_pluralize_cores(sys_nc)). ",
                "However, Julia was started with $(_pluralize_threads(jl_nt)). ",
                "We recommend starting Julia with at least $(_pluralize_threads(sys_nc)) ",
                "to take advantage of Gaius's multithreading algorithms. ",
                "To suppress this warning, set the environment variable ",
                "SUPPRESS_GAIUS_WARNING=true",
            )
            @warn(msg)
        end
    end
    return nothing
end

function _string_to_bool(s::AbstractString)
    b = tryparse(Bool, s)
    if b isa Nothing
        return false
    else
        return b::Bool
    end
end

function _is_suppress_warning()
    s = get(ENV, "SUPPRESS_GAIUS_WARNING", "")
    b = _string_to_bool(s)::Bool
    return b
end

function _pluralize(singular::S, plural::S, n::Integer) where {S <: AbstractString}
    if n == 1
        return singular
    else
        return plural
    end
end

function _pluralize_threads(nt::Integer)
    return "$(nt) $(_pluralize("thread", "threads", nt))"
end

function _pluralize_cores(nc::Integer)
    return "$(nc) $(_pluralize("core", "cores", nc))"
end
