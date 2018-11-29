#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"

namespace vrp::algorithms::construction {

/// Specifies generic insertion heuristic interface.
template<typename Algorithm>
struct InsertionHeuristic {
  InsertionContext insert(const InsertionContext& ctx) const { return static_cast<Algorithm*>(this)->analyze(ctx); }

protected:
  /// Inserts result into context.
  void insert(const InsertionResult& result, InsertionContext& ctx) const {
    result.visit(ranges::overload(
      [&](const InsertionSuccess& success) {
        // perform insertion
        success.route.first->actor = success.actor;
        ctx.registry->use(*success.actor);
        // NOTE assume that activities are sorted by insertion index
        ranges::for_each(success.activities | ranges::view::reverse,
                         [&](const auto& act) { success.route.first->tour.insert(act.first, act.second); });
        ctx.jobs.erase(success.job);
      },
      [&](const InsertionFailure& failure) {
        ranges::for_each(ctx.jobs, [&](const auto& job) { ctx.unassigned[job] = failure.constraint; });
        ctx.jobs.clear();
      }));
  }
};
}
