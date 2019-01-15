#pragma once

#include <chrono>

namespace vrp::utils {

/// Measures execution time of given function.
template<typename TimeT = std::chrono::milliseconds>
struct measure {
  template<typename F, typename... Args>
  static typename TimeT::rep execution(F&& func, Args&&... args) {
    auto start = std::chrono::system_clock::now();
    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);
    return duration.count();
  }
};
}