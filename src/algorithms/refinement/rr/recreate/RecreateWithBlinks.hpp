#pragma once

#include "algorithms/construction/heuristics/BlinkInsertion.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "algorithms/refinement/extensions/RestoreInsertionContext.hpp"
#include "models/Solution.hpp"

namespace vrp::algorithms::refinement {

/// Recreates solution using insertion with blinks heuristic.
struct RecreateWithBlinks final {
  void operator()(const RefinementContext& ctx, models::Solution& sln) const {
    using namespace vrp::algorithms::construction;

    //    auto insertionCtx = restore_insertion_context{}(ctx, sln);
    //    auto evaluator = InsertionEvaluator{ctx.problem->transport, ctx.problem->activity};
    //
    //    auto newCtx = BlinkInsertion<>{evaluator}.operator()(insertionCtx);
    //
    //    sln.routes.clear();
    //    ranges::for_each(newCtx.routes, [](const auto& pair) {
    //
    //    });
    //
    //    sln.unassigned = std::move(newCtx.unassigned);
  }
};
}