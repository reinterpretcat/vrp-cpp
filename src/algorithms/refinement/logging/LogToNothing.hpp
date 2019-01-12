#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"

namespace vrp::algorithms::refinement {

/// Dummy logging which does nothing.
struct log_to_nothing final {
  /// Called when context is created.
  void operator()(const RefinementContext& ctx) const {}

  /// Called when new individuum is discovered.
  void operator()(const RefinementContext& ctx, const models::EstimatedSolution& individuum, bool accepted) const {}

  /// Called when search is completed
  void operator()(const RefinementContext& ctx, int generation) const {}
};
}