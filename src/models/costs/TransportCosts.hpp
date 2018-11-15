#pragma once

#include "models/common/Cost.hpp"
#include "models/common/Distance.hpp"
#include "models/common/Duration.hpp"
#include "models/common/Location.hpp"
#include "models/common/Timestamp.hpp"
#include "models/problem/Actor.hpp"

namespace vrp::models::costs {

/// Provides the way to get routing information for specific locations.
struct TransportCosts {
  /// Returns transport cost between two locations.
  virtual common::Cost cost(const problem::Actor& actor,
                            const common::Location& from,
                            const common::Location& to,
                            const common::Timestamp& departure) const {
    auto dist = distance(actor, from, to, departure);
    auto dur = duration(actor, from, to, departure);
    return dist * (actor.driver->costs.perDistance + actor.vehicle->costs.perDistance) +
      dur * (actor.driver->costs.perDrivingTime + actor.vehicle->costs.perDrivingTime);
  }

  /// Returns transport time between two locations.
  virtual common::Duration duration(const problem::Actor& actor,
                                    const common::Location& from,
                                    const common::Location& to,
                                    const common::Timestamp& departure) const = 0;

protected:
  /// Returns distance between two locations.
  virtual common::Distance distance(const problem::Actor& actor,
                                    const common::Location& from,
                                    const common::Location& to,
                                    const common::Timestamp& departure) const = 0;
};

}  // namespace vrp::models::problem