#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"

#include <chrono>

namespace vrp::algorithms::refinement {

/// Dummy logging which does nothing.
struct log_to_nothing final {
  /// Called when search is started.
  void operator()(const RefinementContext& ctx, std::chrono::milliseconds time) const {}

  /// Called when new individuum is discovered.
  void operator()(const RefinementContext& ctx, const models::EstimatedSolution& individuum, bool accepted) const {}

  /// Called when search is ended within best solution.
  void operator()(const RefinementContext& ctx, const models::EstimatedSolution& best, std::chrono::milliseconds time) {
  }
};
}