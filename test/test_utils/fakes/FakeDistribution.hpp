#pragma once

#include <vector>

namespace vrp::test {

/// Provides the way to generate predefined files.
template<typename T>
struct FakeDistribution {
  std::vector<T> values = {};
  std::size_t index = 0;
  T operator()(T min, T max) {
    assert(index < values.size());
    auto value = values[index++];
    assert(value >= min && value <= max);
    return value;
  }
};
}