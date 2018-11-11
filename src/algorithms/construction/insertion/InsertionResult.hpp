#pragma once

#include <range/v3/utility/variant.hpp>

namespace vrp::algorithms::construction {

/// Specifies insertion result needed to insert job into tour.
struct InsertionSuccess final {};

/// Specifies insertion failure.
struct InsertionFailure final {};

using InsertionResult = ranges::variant<InsertionSuccess, InsertionFailure>;


}  // namespace vrp::algorithms::construction
