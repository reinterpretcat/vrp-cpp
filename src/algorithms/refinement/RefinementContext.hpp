#pragma once

#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "models/common/Cost.hpp"
#include "utils/Random.hpp"

#include <memory>
#include <set>
#include <vector>

namespace vrp::algorithms::refinement {

/// Contains information needed to perform refinement.
struct RefinementContext final {
  /// Specifies individuum type.
  using Individuum = std::shared_ptr<std::pair<models::common::Cost, models::Solution>>;

  /// Original problem.
  std::shared_ptr<const models::Problem> problem;

  /// Random generator.
  std::shared_ptr<utils::Random> random;

  /// Specifies jobs which should not be affected.
  std::shared_ptr<const std::set<models::problem::Job, models::problem::compare_jobs>> locked;

  /// Specifies sorted collection discovered and accepted solutions with their cost.
  std::shared_ptr<std::vector<Individuum>> population;

  /// Specifies refinement generation (or iteration).
  int generation;
};
}