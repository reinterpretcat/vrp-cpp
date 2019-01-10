#pragma once

#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "models/common/Cost.hpp"

namespace vrp::algorithms::objectives {

/// Specifies objective function.
struct ObjectiveFunction {
  /// Specifies actual cost and penalty.
  using Result = std::pair<models::common::Cost, models::common::Cost>;

  /// Estimates solution returning total cost and included penalty.
  virtual Result operator()(const models::Problem& problem, const models::Solution& sln) const = 0;

  virtual ~ObjectiveFunction() = default;
};
}