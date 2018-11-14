#pragma once

#include "models/costs/ActivityCosts.hpp"

#include <algorithm>

namespace vrp::test {

struct TestActivityCosts final : public models::costs::ActivityCosts {
  /// Returns operation time spent to perform activity.
  models::common::Duration duration(const models::problem::Actor& actor,
                                    const models::solution::Activity& activity,
                                    models::common::Timestamp arrival) const override {
    return estimate<models::common::Duration>(activity);
  }

  /// Returns cost to perform activity.
  models::common::Cost cost(const models::problem::Actor& actor,
                            const models::solution::Activity& activity,
                            models::common::Timestamp arrival) const override {
    return estimate<models::common::Cost>(activity);
  }

private:
  template<typename T>
  T estimate(const models::solution::Activity& activity) const {
    return activity.location < 0 ? activity.location * -1 : activity.location;
  }
};
}
