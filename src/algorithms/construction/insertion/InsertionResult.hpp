#pragma once

namespace vrp::algorithms::construction {

struct InsertionResult {
  /// Specifies insertion result needed to insert job into tour.
  struct Success final {};

  /// Specifies insertion failure.
  struct Failure final {};

  InsertionResult() = delete;
};


}  // namespace vrp::algorithms::construction
