cmake_minimum_required(VERSION 3.20)

project(deeplob-serve-cpp
    VERSION 0.1.0
    DESCRIPTION "ONNX Runtime C++ inference engine for deeplob-serve's exported models"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

include(cmake/Warnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/OnnxRuntime.cmake)

option(DEEPLOB_BUILD_TESTS "Build GoogleTest-based unit tests" ON)
option(DEEPLOB_BUILD_BENCHMARKS "Build Google Benchmark microbenchmarks" OFF)

if(DEEPLOB_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

if(DEEPLOB_BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

add_subdirectory(smoke)
add_subdirectory(inference)
if(DEEPLOB_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
