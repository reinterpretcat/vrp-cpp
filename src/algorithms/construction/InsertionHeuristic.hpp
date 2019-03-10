#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/extensions/problem/Helpers.hpp"

#include <algorithm>
#include <chrono>
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
    while (!newCtx.solution->required.empty()) {
      newCtx.problem->constraint->accept(*newCtx.solution);
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
        ctx.solution->registry->use(success.context.route->actor);
        ctx.solution->routes.insert(success.context);

        // NOTE assume that activities are sorted by insertion index
        ranges::for_each(success.activities,
                         [&](const auto& act) { success.context.route->tour.insert(act.first, act.second + 1); });

        // fast erase job from vector
        std::iter_swap(
          std::find_if(ctx.solution->required.begin(),
                       ctx.solution->required.end(),
                       [&](const auto& job) { return models::problem::is_the_same_jobs{}(job, success.job); }),
          ctx.solution->required.end() - 1);
        ctx.solution->required.erase(ctx.solution->required.end() - 1);

        ctx.problem->constraint->accept(success.context);
      },
      [&](InsertionFailure& failure) {
        ranges::for_each(ctx.solution->required,
                         [&](const auto& job) { ctx.solution->unassigned[job] = failure.constraint; });
        ctx.solution->required.clear();
      }));

    ctx.progress.completeness =
      std::max(0.5, 1 - static_cast<double>(ctx.solution->required.size()) / ctx.progress.total);
  }

  const Evaluator evaluator_;
};
}
