#pragma once

#include <range/v3/utility/variant.hpp>

namespace vrp::algorithms::construction {

/// Specifies insertion result needed to insert job into tour.
struct InsertionSuccess final {};

/// Specifies insertion failure.
struct InsertionFailure final {
  /// Failed constraint code
  int constraint;
};

/// Specifies all possible insertion results.
using InsertionResult = ranges::variant<InsertionSuccess, InsertionFailure>;

//// Specifies hard activity constraint check status.
enum class ConstraintStatus {
  Fulfilled,
  NotFulfilledContinue,
  NotFulfilledBreak
};

}  // namespace vrp::algorithms::construction
