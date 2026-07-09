# Warnings-as-errors setup, shared by every target via deeplob_apply_warnings().
#
# -Werror is opt-in via DEEPLOB_WARNINGS_AS_ERRORS so local iteration on a half-written
# milestone isn't blocked by warnings CI will still catch; CI always enables it. Mirrors
# liquibook-x's cmake/Warnings.cmake exactly (same reasoning, same flag set), renamed to this
# project's own prefix.

option(DEEPLOB_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

function(deeplob_apply_warnings target)
    target_compile_options(${target} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
    )
    if(DEEPLOB_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
endfunction()
