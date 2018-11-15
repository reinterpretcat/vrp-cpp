#pragma once

#include "models/common/Cost.hpp"
#include "models/common/Duration.hpp"
#include "models/problem/Actor.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Tour.hpp"

#include <algorithm>

namespace vrp::models::costs {

/// Provides the way to get cost information for specific activities.
struct ActivityCosts {
  /// Returns cost to perform activity.
  virtual common::Cost cost(const models::problem::Actor& actor,
                            const models::solution::Activity& activity,
                            models::common::Timestamp arrival) const {
    auto waiting = activity.time.start > arrival ? activity.time.start - arrival : common::Timestamp{0};
    auto service = duration(actor, activity, arrival);

    return waiting * (actor.driver->costs.perWaitingTime + actor.vehicle->costs.perWaitingTime) +
      service * (actor.driver->costs.perServiceTime + actor.vehicle->costs.perServiceTime);
  }

  /// Returns operation time spent to perform activity.
  virtual common::Duration duration(const models::problem::Actor& actor,
                                    const models::solution::Activity& activity,
                                    models::common::Timestamp arrival) const {
    return activity.duration;
  };

  virtual ~ActivityCosts() = default;
};
}
