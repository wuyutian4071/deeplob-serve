add_library(deeplob_inference STATIC inference_engine.cpp)
target_include_directories(deeplob_inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(deeplob_inference PUBLIC onnxruntime::onnxruntime)
deeplob_apply_warnings(deeplob_inference)
deeplob_apply_sanitizers(deeplob_inference)

if(DEEPLOB_BUILD_TESTS)
    add_executable(deeplob_inference_tests tests/test_inference_engine.cpp)
    target_link_libraries(deeplob_inference_tests PRIVATE deeplob_inference GTest::gtest_main)
    target_compile_definitions(deeplob_inference_tests PRIVATE
        DEEPLOB_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    )
    deeplob_apply_warnings(deeplob_inference_tests)
    deeplob_apply_sanitizers(deeplob_inference_tests)

    include(GoogleTest)
    gtest_discover_tests(deeplob_inference_tests)
endif()
