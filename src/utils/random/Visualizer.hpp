#ifndef VRP_UTILS_RANDOM_VISUALIZER_HPP
#define VRP_UTILS_RANDOM_VISUALIZER_HPP

#include "runtime/Config.hpp"

#include <iomanip>
#include <map>

namespace vrp {
namespace utils {

/// Visualizes generated numbers using histogram.
struct visualize_generator {
  template<typename Generator>
  void operator()(Generator& generator) {
    std::map<int, int> hist{};
    for (int n = 0; n < 10000; ++n) {
      auto value = generator();
      ++hist[value];
    }
    for (auto p : hist) {
      std::cout << std::setw(2) << p.first << ' ' << std::string(p.second / 100, '*') << '\n';
    }
  }
};

}  // namespace utils
}  // namespace vrp

#endif  // VRP_UTILS_RANDOM_VISUALIZER_HPP
