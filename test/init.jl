@testset "init" begin
    @testset "_print_num_threads_warning" begin
        withenv("SUPPRESS_GAIUS_WARNING" => "false") do
            @test_logs (:warn, "The system has 16 threads. However, Julia was started with 1 thread. We recommend starting Julia with 16 threads to take advantage of Gaius's multithreading algorithms. To suppress this warning, set the environment variable SUPPRESS_GAIUS_WARNING=true") match_mode=:any Gaius._print_num_threads_warning(16, 1)
        end
    end

    @testset "_string_to_bool" begin
        @test !Gaius._string_to_bool("")
        @test !Gaius._string_to_bool("foobarbaz")
        @test !Gaius._string_to_bool("false")
        @test !Gaius._string_to_bool("0")
        @test Gaius._string_to_bool("true")
        @test Gaius._string_to_bool("1")
    end

    @testset "_is_suppress_warning" begin
        withenv("SUPPRESS_GAIUS_WARNING" => nothing) do
            @test !Gaius._is_suppress_warning()
        end
        withenv("SUPPRESS_GAIUS_WARNING" => "") do
            @test !Gaius._is_suppress_warning()
        end
        withenv("SUPPRESS_GAIUS_WARNING" => "false") do
            @test !Gaius._is_suppress_warning()
        end
        withenv("SUPPRESS_GAIUS_WARNING" => "0") do
            @test !Gaius._is_suppress_warning()
        end
        withenv("SUPPRESS_GAIUS_WARNING" => "true") do
            @test Gaius._is_suppress_warning()
        end
        withenv("SUPPRESS_GAIUS_WARNING" => "1") do
            @test Gaius._is_suppress_warning()
        end
    end

    @testset "_pluralize" begin
        @test Gaius._pluralize("foo", "bar", 0) == "bar"
        @test Gaius._pluralize("foo", "bar", 1) == "foo"
        @test Gaius._pluralize("foo", "bar", 2) == "bar"
        @test Gaius._pluralize("foo", "bar", 3) == "bar"
    end

    @testset "_pluralize_nt" begin
        @test Gaius._pluralize_nt(0) == "0 threads"
        @test Gaius._pluralize_nt(1) == "1 thread"
        @test Gaius._pluralize_nt(2) == "2 threads"
        @test Gaius._pluralize_nt(3) == "3 threads"
    end
end
