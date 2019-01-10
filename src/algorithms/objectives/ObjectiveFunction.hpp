#pragma once

#include "models/Solution.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"

namespace vrp::algorithms::objectives {

/// Specifies objective function.
struct ObjectiveFunction {
  /// Specifies actual cost and penalty.
  using Result = std::pair<models::common::Cost, models::common::Cost>;

  /// Estimates solution returning total cost and included penalty.
  virtual Result operator()(const models::Solution& sln,
                            const models::costs::ActivityCosts& activity,
                            const models::costs::TransportCosts& transport) const = 0;

  virtual ~ObjectiveFunction() = default;
};
}