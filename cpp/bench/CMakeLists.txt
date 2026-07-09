add_executable(deeplob_bench_inference_latency bench_inference_latency.cpp)
target_link_libraries(deeplob_bench_inference_latency PRIVATE deeplob_inference benchmark::benchmark)
target_compile_definitions(deeplob_bench_inference_latency PRIVATE
    DEEPLOB_BENCH_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)
deeplob_apply_warnings(deeplob_bench_inference_latency)
# Deliberately no deeplob_apply_sanitizers() here -- ASan/UBSan overhead would make the
# reported latency/throughput numbers meaningless, same reasoning as liquibook-x's own
# bench/CMakeLists.txt.
