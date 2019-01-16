#pragma once

#include <chrono>

namespace vrp::utils {

/// Measures execution time of given function.
template<typename TimeT = std::chrono::milliseconds>
struct measure {
  template<typename Func, typename... Args>
  static typename TimeT::rep execution(Func&& func, Args&&... args) {
    auto start = std::chrono::system_clock::now();
    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);
    return duration.count();
  }

  template<typename Func, typename Logger>
  static auto execution_return_result(Func&& func, Logger&& logger) {
    auto start = std::chrono::system_clock::now();
    auto result = func();
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);
    logger(result, duration);
    return std::move(result);
  }
};
}