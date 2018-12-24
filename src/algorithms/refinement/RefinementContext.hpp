#pragma once

#include "models/Problem.hpp"
#include "utils/Random.hpp"

#include <memory>

namespace vrp::algorithms::refinement {

/// Contains information needed to perform refinement.
struct RefinementContext final {
  /// Original problem.
  std::shared_ptr<const models::Problem> problem;

  /// Random generator.
  std::shared_ptr<utils::Random> random;

  // TODO: locked jobs?
  // std::vector<models::problem::Job> locked;
};
}