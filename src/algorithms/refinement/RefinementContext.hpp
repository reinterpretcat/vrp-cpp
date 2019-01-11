#pragma once

#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "utils/Random.hpp"

#include <memory>
#include <set>
#include <vector>

namespace vrp::algorithms::refinement {

/// Contains information needed to perform refinement.
struct RefinementContext final {
  /// Specifies population type.
  using Population = std::vector<models::EstimatedSolution>;

  /// Original problem.
  std::shared_ptr<const models::Problem> problem;

  /// Random generator.
  std::shared_ptr<utils::Random> random;

  /// Specifies jobs which should not be affected.
  std::shared_ptr<const std::set<models::problem::Job, models::problem::compare_jobs>> locked;

  /// Specifies sorted collection discovered and accepted solutions with their cost.
  std::shared_ptr<Population> population;

  /// Specifies refinement generation (or iteration).
  int generation;
};
}