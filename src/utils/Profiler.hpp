#ifndef VRP_UTILS_PROFILER_HPP
#define VRP_UTILS_PROFILER_HPP

#include <nvToolsExt.h>
#include <utility>

namespace vrp {
namespace utils {

/// Provides the way to execute function using profiler time range.
struct profile {
  template<typename F, typename ...Args>
  static void execution(const char *tag, F &&func, Args &&... args) {
    auto id = nvtxRangeStart(tag);
    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
    nvtxRangeEnd(id);
  }
};

}
}

#endif //VRP_UTILS_PROFILER_HPP
