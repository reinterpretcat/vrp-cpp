#pragma once

#include <limits>

namespace vrp::models::common {

/// Represents cost value.
using Cost = double;

/// Specifies no cost constant.
inline Cost NoCost = std::numeric_limits<Cost>::max();

/// Represents actual cost and penalty.
struct ObjectiveCost final {
  /// Actual cost.
  Cost actual;
  /// Penalty cost.
  Cost penalty;

  /// Returns total cost.
  Cost total() const { return actual + penalty; }
};
}