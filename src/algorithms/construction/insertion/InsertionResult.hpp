#pragma once

#include <variant>

namespace vrp::algorithms::construction {

struct InsertionResult {
  /// Specifies insertion result needed to insert job into tour.
  struct Success final {};

  /// Specifies insertion failure.
  struct Failure final {};

  InsertionResult() = delete;

  using Variant = std::variant<InsertionResult::Success, InsertionResult::Failure>;
};


}  // namespace vrp::algorithms::construction
