#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "algorithms/refinement/extensions/SelectBestSolution.hpp"
#include "models/Solution.hpp"
#include "models/common/Cost.hpp"

#include <cmath>

namespace vrp::algorithms::refinement {

/// Greedy acceptance which accepts only better solutions.
template<typename Selector = select_best_solution>
struct GreedyAcceptance final {
  /// Specifies selector which selects solution from context.
  Selector selector;

  bool operator()(const RefinementContext& ctx, const models::EstimatedSolution& individuum) const {
    return individuum.second.total() < selector(ctx).second.total();
  }
};
}
