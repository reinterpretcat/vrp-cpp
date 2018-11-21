#pragma once

#include "models/common/Timestamp.hpp"
#include "models/solution/Actor.hpp"
#include "models/solution/Route.hpp"
#include "models/solution/Tour.hpp"

#include <memory>
#include <range/v3/utility/variant.hpp>

namespace vrp::algorithms::construction {

/// Specifies insertion result needed to insert job into tour.
struct InsertionSuccess final {
  /// Specifies index where activity has to be inserted.
  size_t index;

  /// Specifies activity which has to be inserted.
  models::solution::Tour::Activity activity;

  /// Specifies actor which should be used.
  models::solution::Route::Actor actor;

  /// Specifies new vehicle departure time.
  models::common::Timestamp departure;
};

/// Specifies insertion failure.
struct InsertionFailure final {
  /// Failed constraint code
  int constraint;
};

/// Specifies all possible insertion results.
using InsertionResult = ranges::variant<InsertionSuccess, InsertionFailure>;

}  // namespace vrp::algorithms::construction
