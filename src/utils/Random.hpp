#pragma once

#include <chrono>
#include <random>
#include <type_traits>

namespace vrp::utils {

/// Provides the way to use randomized values in generic way.
class Random final {
public:
  explicit Random(std::random_device device) : generator_(device()) {}

  /// Produces random value, uniformly distributed on the closed interval [min, max]
  template<typename T>
  T uniform(T min, T max) {
    using Dist = typename std::conditional<std::is_integral<T>::value,
                                           std::uniform_int_distribution<T>,
                                           std::uniform_real_distribution<T>>::type;
    return Dist(min, max)(generator_);
  }

private:
  std::mt19937 generator_;
};
}