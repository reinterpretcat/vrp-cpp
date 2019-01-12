#pragma once

#include "algorithms/refinement/RefinementContext.hpp"

namespace vrp::algorithms::refinement {

/// Stops when maximum amount of iterations is reached.
struct MaxIterationCriteria final {
  int maxIterations = 10;

  /// Returns true if algorithm should be terminated.
  bool operator()(const RefinementContext& ctx, const models::EstimatedSolution&, bool accepted) const {
    return ctx.generation > maxIterations;
  }
};
}