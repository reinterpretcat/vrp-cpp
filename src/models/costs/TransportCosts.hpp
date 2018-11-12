#pragma once

#include "models/common/Cost.hpp"
#include "models/common/Duration.hpp"
#include "models/common/Location.hpp"
#include "models/common/Timestamp.hpp"
#include "models/problem/Actor.hpp"

namespace vrp::models::behavioral {

/// Provides the way to get routing information for specific locations.
struct TransportCosts {
  /// Returns transport time between two locations.
  virtual common::Duration duration(const problem::Actor& actor,
                                    const common::Location& from,
                                    const common::Location& to,
                                    const common::Timestamp& departure) = 0;

  /// Returns transport cost between two locations.
  virtual common::Cost cost(const problem::Actor& actor,
                            const common::Location& from,
                            const common::Location& to,
                            const common::Timestamp& departure) = 0;
};

}  // namespace vrp::models::problem