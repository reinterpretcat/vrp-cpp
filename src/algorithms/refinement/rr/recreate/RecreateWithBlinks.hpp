#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/heuristics/BlinkInsertion.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "algorithms/refinement/extensions/RestoreInsertionContext.hpp"
#include "models/Solution.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::refinement {

/// Recreates solution using insertion with blinks heuristic.
struct RecreateWithBlinks final {
  /// Creates a new solution from contexts.
  models::Solution operator()(const RefinementContext& rCtx, const construction::InsertionContext& iCtx) const {
    using namespace vrp::algorithms::construction;
    using namespace ranges;
    using ConstRoute = std::shared_ptr<const models::solution::Route>;

    auto evaluator = InsertionEvaluator{rCtx.problem->transport, rCtx.problem->activity};
    auto resultCtx = BlinkInsertion<>{evaluator}.operator()(iCtx);

    return models::Solution{
      resultCtx.registry,
      resultCtx.routes | view::transform([](const auto& p) -> ConstRoute { return p.first; }) | to_vector,
      resultCtx.unassigned};
  }
};
}
