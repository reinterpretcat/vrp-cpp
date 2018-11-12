#pragma once

#include "models/common/Cost.hpp"
#include "models/common/Duration.hpp"
#include "models/problem/Actor.hpp"
#include "models/solution/Activity.hpp"

namespace vrp::models::behavioral {

/// Provides the way to get cost information for specific activities.
struct ActivityCosts {
  /// Returns operation time spent to perform activity.
  virtual common::Duration duration(const models::solution::Tour::Activity& activity,
                                    const models::problem::Actor& actor,
                                    models::common::Timestamp arrival) const = 0;

  /// Returns cost to perform activity.
  virtual common::Cost cost(const models::solution::Tour::Activity& activity,
                            const models::problem::Actor& actor,
                            models::common::Timestamp arrival) const = 0;

  virtual ~Costs() = default;
};
}
