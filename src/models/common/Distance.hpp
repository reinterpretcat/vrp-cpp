#pragma once

#include <cstdint>
#include <limits>

namespace vrp::models::common {

/// Represents a distance.
using Distance = double;

/// Specifies no distance constant.
inline Distance NoDistance = std::numeric_limits<Distance>::max();
}  // namespace vrp::models::common
