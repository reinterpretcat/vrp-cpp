#pragma once

#include <cstdint>
#include <limits>

namespace vrp::models::common {

/// Represents a time unit.
using Timestamp = double;

/// Specifies maximum value of timestamp.
inline Timestamp MaxTime = std::numeric_limits<common::Timestamp>::max();

}  // namespace vrp::models::common
