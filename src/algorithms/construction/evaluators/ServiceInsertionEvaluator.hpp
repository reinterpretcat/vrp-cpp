#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/evaluators/EvaluationContext.hpp"
#include "models/common/Cost.hpp"
#include "models/common/TimeWindow.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/problem/Service.hpp"
#include "utils/extensions/Ranges.hpp"

#include <numeric>
#include <range/v3/all.hpp>
#include <tuple>
#include <utility>

namespace vrp::algorithms::construction {

struct ServiceInsertionEvaluator final {
  /// Evaluates service insertion possibility.
  InsertionResult evaluate(const std::shared_ptr<const models::problem::Service>& service,
                           const InsertionRouteContext& ctx,
                           const InsertionConstraint& constraint,
                           const InsertionProgress& progress) const {
    auto job = models::problem::as_job(service);

    // check hard constraints on route level.
    auto error = constraint.hard(ctx, job);
    if (error.has_value()) return {ranges::emplaced_index<1>, InsertionFailure{error.value()}};

    return analyze(job, *service, ctx, constraint, progress);
  }

private:
  using Activity = models::solution::Tour::Activity;

  /// Analyzes tour trying to find best insertion index.
  InsertionResult analyze(const models::problem::Job& job,
                          const models::problem::Service& service,
                          const InsertionRouteContext& ctx,
                          const InsertionConstraint& constraint,
                          const InsertionProgress& progress) const {
    using namespace ranges;
    using namespace vrp::models;
    using ActivityType = solution::Activity::Type;

    const auto& route = *ctx.route;
    auto activity = std::make_shared<solution::Activity>(solution::Activity{ActivityType::Job, {}, {}, job});

    // calculate additional costs on route level.
    auto routeCosts = constraint.soft(ctx, job);

    // form route legs from a new route view.
    auto tour = view::concat(view::single(route.start), route.tour.activities(), view::single(route.end));
    auto legs = view::zip(tour | view::sliding(2), view::iota(static_cast<size_t>(0)));
    auto evalCtx = EvaluationContext::empty(progress.bestCost);
    auto pred = [](const EvaluationContext& ctx) { return !ctx.isStopped; };

    // 1. analyze route legs
    auto result = utils::accumulate_while(legs, evalCtx, pred, [&](const auto& out, const auto& view) {
      auto [items, index] = view;
      auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
      auto actCtx = InsertionActivityContext{index, prev, activity, next};

      // 2. analyze service details
      return utils::accumulate_while(view::all(service.details), out, pred, [&](const auto& in1, const auto& detail) {
        // TODO check whether tw is empty
        // 3. analyze detail time windows
        return utils::accumulate_while(view::all(detail.times), in1, pred, [&](const auto& in2, const auto& time) {
          activity->detail = {detail.location.value_or(actCtx.prev->detail.location), detail.duration, time};
          // check hard activity constraint
          auto status = constraint.hard(ctx, actCtx);
          if (status.has_value()) return EvaluationContext::fail(status.value(), in2);

          auto totalCosts = routeCosts + constraint.soft(ctx, actCtx);
          return totalCosts < in2.cost
            ? EvaluationContext::success(actCtx.index, totalCosts, {activity->detail.location, detail.duration, time})
            : EvaluationContext::skip(in2);
        });
      });
    });

    activity->detail = result.detail;

    return result.isSuccess()
      ? make_result_success({result.cost, activity->job.value(), {{activity, result.index}}, ctx})
      : make_result_failure(result.code);
  }
};

}  // namespace vrp::algorithms::construction
