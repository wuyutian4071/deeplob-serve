#pragma once

#include <string_view>

namespace deeplob {

// Proves the whole C++ toolchain (compile, link, GoogleTest via FetchContent, ctest
// discovery, sanitizer builds, CI) works end to end before any real module exists -- mirrors
// liquibook-x's own smoke/ pattern. Real modules (inference/, bench/) replace this at M8.
[[nodiscard]] std::string_view version() noexcept;

} // namespace deeplob
