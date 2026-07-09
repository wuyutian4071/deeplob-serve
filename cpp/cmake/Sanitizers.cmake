# Sanitizer setup, shared by every target via deeplob_apply_sanitizers().
#
# ASan and UBSan only -- unlike liquibook-x, this inference engine has no planned concurrent
# component (batch inference is single-threaded; there's no lock-free structure or thread
# handoff in this project's spec), so TSan plumbing would be unused complexity, not
# forward-looking infrastructure.

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

function(deeplob_apply_sanitizers target)
    set(_sanitizers "")
    if(ENABLE_ASAN)
        list(APPEND _sanitizers "address")
    endif()
    if(ENABLE_UBSAN)
        list(APPEND _sanitizers "undefined")
    endif()

    if(_sanitizers)
        list(JOIN _sanitizers "," _sanitizers_csv)
        target_compile_options(${target} PRIVATE
            -fsanitize=${_sanitizers_csv}
            -fno-omit-frame-pointer
        )
        target_link_options(${target} PRIVATE -fsanitize=${_sanitizers_csv})
    endif()
endfunction()
