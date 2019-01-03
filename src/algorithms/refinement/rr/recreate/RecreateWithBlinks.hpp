#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "algorithms/refinement/extensions/RestoreInsertionContext.hpp"
#include "models/Solution.hpp"

namespace vrp::algorithms::refinement {

/// Recreates solution using insertion with blinks heuristic.
struct RecreateWithBlinks final {
  void operator()(const RefinementContext& ctx, models::Solution& sln) const {
    auto insertionCtx = restore_insertion_context{}(ctx, sln);

    // TODO run insertion heuristic
  }
};
}