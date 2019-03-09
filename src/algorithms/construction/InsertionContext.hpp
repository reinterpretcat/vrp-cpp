#pragma once

#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionSolutionContext.hpp"
#include "models/Problem.hpp"
#include "models/solution/Registry.hpp"
#include "utils/Random.hpp"

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Contains information needed to performed insertions in solution.
struct InsertionContext final {
  /// Solution progress.
  InsertionProgress progress;

  /// Original problem.
  std::shared_ptr<const models::Problem> problem;

  /// Solution context.
  std::shared_ptr<InsertionSolutionContext> solution;

  /// Keeps track of used resources.
  std::shared_ptr<models::solution::Registry> registry;

  /// Random generator.
  std::shared_ptr<utils::Random> random;
};
}