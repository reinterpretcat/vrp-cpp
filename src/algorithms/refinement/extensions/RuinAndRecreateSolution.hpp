#pragma once

#include "algorithms/refinement/rar/recreate/RecreateWithBlinks.hpp"
#include "algorithms/refinement/rar/ruin/RemoveAdjustedString.hpp"

namespace vrp::algorithms::refinement {

/// Ruins and recreates solution.
template<typename Ruin = RemoveAdjustedString, typename Recreate = RecreateWithBlinks>
struct ruin_and_recreate_solution final {
  models::EstimatedSolution operator()(const RefinementContext& ctx, const models::EstimatedSolution& sln) const {
    // TODO how to pass settings?
    auto newSln = std::make_shared<models::Solution>(Recreate{}(ctx, Ruin{}(ctx, *sln.first)));
    auto cost = ctx.problem->objective->operator()(*newSln, *ctx.problem->activity, *ctx.problem->transport);

    return {newSln, cost};
  }
};
}