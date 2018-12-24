#pragma once

#include <chrono>
#include <random>

namespace vrp::utils {

/// Provides the way to use randomized values in generic way.
class Random final {
public:
  explicit Random(const std::random_device& device) : generator_(device) {}

private:
  std::mt19937 generator_;
};
}