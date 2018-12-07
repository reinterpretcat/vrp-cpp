#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"

namespace vrp::algorithms::construction {

/// Specifies generic insertion heuristic interface.
template<typename Algorithm>
struct InsertionHeuristic {
  InsertionContext operator()(const InsertionContext& ctx) const {
    return static_cast<const Algorithm*>(this)->insert(ctx);
  }

protected:
  /// Inserts result into context.
  void insert(const InsertionResult& result, InsertionContext& ctx) const {
    result.visit(ranges::overload(
      [&](const InsertionSuccess& success) {
        success.route.first->actor = success.actor;
        success.route.first->start->schedule.departure = success.departure;

        ctx.registry->use(*success.actor);
        ctx.routes[success.route.first] = success.route.second;

        // NOTE assume that activities are sorted by insertion index
        ranges::for_each(success.activities | ranges::view::reverse,
                         [&](const auto& act) { success.route.first->tour.insert(act.first, act.second); });

        ctx.jobs.erase(success.job);
        ctx.constraint->accept(*success.route.first, *success.route.second);
      },
      [&](const InsertionFailure& failure) {
        ranges::for_each(ctx.jobs, [&](const auto& job) { ctx.unassigned[job] = failure.constraint; });
        ctx.jobs.clear();
      }));
  }
};
}
