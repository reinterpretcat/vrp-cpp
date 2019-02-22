#pragma once

#include "algorithms/refinement/recreate/RecreateWithBlinks.hpp"
#include "algorithms/refinement/ruin/RemoveAdjustedString.hpp"

namespace vrp::algorithms::refinement {

/// Ruins and recreates solution.
template<typename Ruin = RemoveAdjustedString, typename Recreate = RecreateWithBlinks>
struct ruin_and_recreate_solution final {
  models::EstimatedSolution operator()(const RefinementContext& ctx, const models::EstimatedSolution& sln) const {
    // TODO how to pass settings?

    auto iCtx = restore_insertion_context{}(ctx, *sln.first);

    Ruin{}(ctx, *sln.first, iCtx);

    auto newSln = std::make_shared<models::Solution>(Recreate{}(ctx, iCtx));

    auto cost = ctx.problem->objective->operator()(*newSln, *ctx.problem->activity, *ctx.problem->transport);

    return {newSln, cost};
  }
};
}