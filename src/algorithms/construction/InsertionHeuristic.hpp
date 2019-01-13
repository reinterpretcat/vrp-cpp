#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"

#include <pstl/execution>
#include <pstl/numeric>

namespace vrp::algorithms::construction {

/// Specifies generic insertion heuristic logic.
/// Evaluator template param specifies the logic responsible for job insertion
/// evaluation: where is job's insertion point.
/// Selector template param specifies the logic which selects job to be inserted.
template<typename Evaluator, typename Selector>
struct InsertionHeuristic {
  explicit InsertionHeuristic(const Evaluator& evaluator) : evaluator_(evaluator) {}

  InsertionContext operator()(const InsertionContext& ctx) const {
    auto newCtx = InsertionContext(ctx);
    auto selector = Selector(ctx);
    while (!newCtx.jobs.empty()) {
      insert(std::transform_reduce(pstl::execution::par,
                                   newCtx.jobs.begin(),
                                   newCtx.jobs.end(),
                                   make_result_failure(),
                                   [&](const auto& acc, const auto& result) { return selector(acc, result); },
                                   [&](const auto& job) { return evaluator_.evaluate(job, newCtx); }),
             newCtx);
    }
    return std::move(newCtx);
  }

private:
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

    ctx.progress.completeness = std::max(0.5, 1 - static_cast<double>(ctx.jobs.size()) / ctx.progress.total);
  }

  const Evaluator evaluator_;
};
}
