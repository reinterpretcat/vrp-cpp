#pragma once

#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "test_utils/models/Validators.hpp"

#include <iostream>

namespace vrp::test {

/// Logs basic information to console and validates.
struct log_and_validate final {
  /// Called when search is started.
  void operator()(const algorithms::refinement::RefinementContext& ctx, std::chrono::milliseconds duration) const {
    logger(ctx, duration);
    validator(*ctx.problem, *ctx.population->front().first);
  }

  /// Called when new individuum is discovered.
  void operator()(const algorithms::refinement::RefinementContext& ctx,
                  const models::EstimatedSolution& individuum,
                  std::chrono::milliseconds duration,
                  bool accepted) const {
    logger(ctx, individuum, duration, accepted);
    validator(*ctx.problem, *individuum.first);
  }

  /// Called when search is ended within best solution.
  void operator()(const algorithms::refinement::RefinementContext& ctx,
                  const models::EstimatedSolution& best,
                  std::chrono::milliseconds duration) {
    logger(ctx, best, duration);
    validator(*ctx.problem, *best.first);
  }

private:
  algorithms::refinement::log_to_console logger = {};
  validate_solution<int, false> validator = {};
};
}