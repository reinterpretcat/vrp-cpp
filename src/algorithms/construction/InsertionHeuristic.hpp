#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "models/extensions/problem/Properties.hpp"

#include <algorithm>
#include <iostream>
#include <pstl/execution>
#include <pstl/numeric>

namespace vrp::algorithms::construction {

/// Specifies generic insertion heuristic logic.
/// Evaluator template param specifies the logic responsible for job insertion
///     evaluation: where is job's insertion point.
/// JobSelector template param specifies the logic responsible for job selection.
/// ResultSelector template param specifies the logic which selects which result to be inserted.
template<typename Evaluator, typename JobSelector, typename ResultSelector>
struct InsertionHeuristic {
  explicit InsertionHeuristic(const Evaluator& evaluator) : evaluator_(evaluator) {}

  InsertionContext operator()(const InsertionContext& ctx) const {
    auto newCtx = InsertionContext(ctx);
    auto rSelector = ResultSelector(ctx);
    while (!newCtx.jobs.empty()) {
      auto [begin, end] = JobSelector{}(newCtx);
      auto result = std::transform_reduce(pstl::execution::seq,
                                          begin,
                                          end,
                                          make_result_failure(),
                                          [&](const auto& acc, const auto& result) { return rSelector(acc, result); },
                                          [&](const auto& job) { return evaluator_.evaluate(job, newCtx); });
      insert(result, newCtx);
    }
    return std::move(newCtx);
  }

private:
  /// Inserts result into context.
  void insert(InsertionResult& result, InsertionContext& ctx) const {
    result.visit(ranges::overload(
      [&](InsertionSuccess& success) {
        std::cout << "left " << ctx.jobs.size() << " "
                  << "insert " << success.activities.front().first->detail.location << " at index "
                  << success.activities.front().second << " into route with "
                  << success.context.route->tour.sizes().second << " activities "
                  << success.context.route->actor->vehicle->id << " total routes: " << ctx.routes.size() << "\n";

        ctx.registry->use(*success.context.route->actor);
        ctx.routes.insert(success.context);

        // NOTE assume that activities are sorted by insertion index
        ranges::for_each(success.activities | ranges::view::reverse,
                         [&](const auto& act) { success.context.route->tour.insert(act.first, act.second); });

        // TODO restore logic once bug is found
        // fast erase job from vector
        ctx.jobs.erase(std::find_if(ctx.jobs.begin(), ctx.jobs.end(), [&](const auto& job) {
          const static auto getter = models::problem::get_job_id{};
          return getter(job) == getter(success.job);
        }));
        // ctx.jobs.erase(ctx.jobs.end() - 1);

        ctx.constraint->accept(success.context);
      },
      [&](InsertionFailure& failure) {
        // TODO handle properly
        ranges::for_each(ctx.jobs, [&](const auto& job) { ctx.unassigned[job] = failure.constraint; });
        ctx.jobs.clear();
      }));

    ctx.progress.completeness = std::max(0.5, 1 - static_cast<double>(ctx.jobs.size()) / ctx.progress.total);
  }

  const Evaluator evaluator_;
};
}
