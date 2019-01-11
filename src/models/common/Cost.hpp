#pragma once

namespace vrp::models::common {

/// Represents cost value.
using Cost = double;

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