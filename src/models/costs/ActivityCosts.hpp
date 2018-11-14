#pragma once

#include "models/common/Cost.hpp"
#include "models/common/Duration.hpp"
#include "models/problem/Actor.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Tour.hpp"

namespace vrp::models::costs {

/// Provides the way to get cost information for specific activities.
struct ActivityCosts {
  /// Returns operation time spent to perform activity.
  virtual common::Duration duration(const models::problem::Actor& actor,
                                    const models::solution::Activity& activity,
                                    models::common::Timestamp arrival) const = 0;

  /// Returns cost to perform activity.
  virtual common::Cost cost(const models::problem::Actor& actor,
                            const models::solution::Activity& activity,
                            models::common::Timestamp arrival) const = 0;

  virtual ~ActivityCosts() = default;
};
}
