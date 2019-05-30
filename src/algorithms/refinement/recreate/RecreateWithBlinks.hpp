#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/heuristics/BlinkInsertion.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "algorithms/refinement/extensions/RestoreInsertionContext.hpp"
#include "models/Solution.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::refinement {

/// Recreates solution using insertion with blinks heuristic.
struct recreate_with_blinks final {
  /// Creates a new solution from contexts.
  models::Solution operator()(const RefinementContext& rCtx, const construction::InsertionContext& iCtx) const {
    using namespace vrp::algorithms::construction;
    using namespace ranges;
    using ConstRoute = std::shared_ptr<const models::solution::Route>;

    auto evaluator = InsertionEvaluator{};
    auto resultCtx = BlinkInsertion<>{evaluator}.operator()(iCtx);

    return models::Solution{
      resultCtx.solution->registry,
      resultCtx.solution->routes | view::transform([](const auto& p) -> ConstRoute { return p.route; }) | to_vector,
      std::move(resultCtx.solution->unassigned)};
  }
};
}
