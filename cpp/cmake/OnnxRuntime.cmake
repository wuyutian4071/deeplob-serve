# Downloads a platform-specific prebuilt ONNX Runtime C++ SDK release and exposes it as an
# imported target `onnxruntime::onnxruntime`.
#
# There is no first-party CMake package for onnxruntime that FetchContent's usual
# GIT_REPOSITORY + add_subdirectory pattern works with -- building onnxruntime from source is
# a multi-hour, heavyweight process, not practical for CI or local iteration. Microsoft's own
# prebuilt release tarballs (headers + a shared library, no build system inside) are the
# standard way C++ consumers get this dependency; FetchContent_Populate (not
# FetchContent_MakeAvailable, which expects a CMakeLists.txt to add_subdirectory) is the
# correct FetchContent entry point for exactly this "just download and unpack" case.
#
# Version pinned to 1.27.0 to match this project's Python-side onnxruntime version exactly
# (src/deeplob's own dependency, see pyproject.toml) -- not a coincidence, deliberately kept
# in sync so the model export (M7) and this C++ inference engine are verified against the
# same onnxruntime release, not two versions that could behave subtly differently.

set(DEEPLOB_ONNXRUNTIME_VERSION "1.27.0")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(_ort_platform "osx-arm64")
    else()
        set(_ort_platform "osx-x86_64")
    endif()
    set(_ort_lib_name "libonnxruntime.dylib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_ort_platform "linux-x64")
    set(_ort_lib_name "libonnxruntime.so")
else()
    message(FATAL_ERROR "No prebuilt ONNX Runtime release known for platform: ${CMAKE_SYSTEM_NAME}")
endif()

set(_ort_archive_name "onnxruntime-${_ort_platform}-${DEEPLOB_ONNXRUNTIME_VERSION}")

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)  # extracted files get extraction-time timestamps
endif()
if(POLICY CMP0169)
    # FetchContent_Populate's low-level (non-MakeAvailable) form is deprecated in newer CMake
    # in favor of add_subdirectory-based content -- doesn't apply here, since onnxruntime's
    # release tarball has no CMakeLists.txt to add_subdirectory (see comment above); OLD keeps
    # this call working without a functional replacement being needed.
    cmake_policy(SET CMP0169 OLD)
endif()

include(FetchContent)
FetchContent_Declare(
    onnxruntime
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${DEEPLOB_ONNXRUNTIME_VERSION}/${_ort_archive_name}.tgz"
)
FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
    FetchContent_Populate(onnxruntime)
endif()

add_library(onnxruntime_imported SHARED IMPORTED GLOBAL)
add_library(onnxruntime::onnxruntime ALIAS onnxruntime_imported)
set_target_properties(onnxruntime_imported PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/${_ort_lib_name}"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

# Neither macOS's dylib IDs nor Linux's .so SONAME point anywhere a built executable finds
# automatically by default -- rpath makes every executable/test binary able to find the
# shared library at its FetchContent-downloaded location without needing it copied alongside
# or installed system-wide. Needed on both platforms, not just macOS -- this was tested and
# verified working on macOS first (the development machine), but Linux (CI's actual runner)
# needs the identical mechanism, not just an assumption it works the same way.
if(APPLE)
    set_target_properties(onnxruntime_imported PROPERTIES IMPORTED_NO_SONAME FALSE)
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${onnxruntime_SOURCE_DIR}/lib")
set(CMAKE_BUILD_RPATH "${onnxruntime_SOURCE_DIR}/lib")
