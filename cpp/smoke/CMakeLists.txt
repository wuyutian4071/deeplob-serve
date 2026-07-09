add_library(deeplob_smoke STATIC version.cpp)
target_include_directories(deeplob_smoke PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
deeplob_apply_warnings(deeplob_smoke)
deeplob_apply_sanitizers(deeplob_smoke)

if(DEEPLOB_BUILD_TESTS)
    add_executable(deeplob_smoke_tests test_version.cpp)
    target_link_libraries(deeplob_smoke_tests PRIVATE deeplob_smoke GTest::gtest_main)
    deeplob_apply_warnings(deeplob_smoke_tests)
    deeplob_apply_sanitizers(deeplob_smoke_tests)

    include(GoogleTest)
    gtest_discover_tests(deeplob_smoke_tests)
endif()
