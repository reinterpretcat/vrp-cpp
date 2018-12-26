#pragma once

#include "models/Problem.hpp"
#include "utils/Random.hpp"

#include <memory>
#include <set>

namespace vrp::algorithms::refinement {

/// Contains information needed to perform refinement.
struct RefinementContext final {
  /// Original problem.
  std::shared_ptr<const models::Problem> problem;

  /// Random generator.
  std::shared_ptr<utils::Random> random;

  /// Specifies jobs which should not be affected.
  std::shared_ptr<std::set<models::problem::Job, models::problem::compare_jobs>> locked;
};
}