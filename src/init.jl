function __init__()
    _print_num_threads_warning()
    return nothing
end

function _print_num_threads_warning()
    sys_nt = Sys.CPU_THREADS
    jl_nt = Threads.nthreads()
    _print_num_threads_warning(sys_nt, jl_nt)
    return nothing
end

function _print_num_threads_warning(sys_nt::Integer, jl_nt::Integer)
    if jl_nt < sys_nt
        if !_is_suppress_warning()
            msg = string(
                "The system has $(_pluralize_nt(sys_nt)). ",
                "However, Julia was started with $(_pluralize_nt(jl_nt)). ",
                "We recommend starting Julia with $(_pluralize_nt(sys_nt)) ",
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

function _pluralize_nt(nt::Integer)
    return "$(nt) $(_pluralize("thread", "threads", nt))"
end
