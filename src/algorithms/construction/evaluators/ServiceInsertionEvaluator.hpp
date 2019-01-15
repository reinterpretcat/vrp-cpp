#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/evaluators/JobInsertionEvaluator.hpp"
#include "algorithms/construction/extensions/Routes.hpp"
#include "models/common/Cost.hpp"
#include "models/common/TimeWindow.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Service.hpp"

#include <numeric>
#include <range/v3/all.hpp>
#include <tuple>
#include <utility>

namespace vrp::algorithms::construction {

struct ServiceInsertionEvaluator final : private JobInsertionEvaluator {
  ServiceInsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                            std::shared_ptr<const models::costs::ActivityCosts> activityCosts) :
    JobInsertionEvaluator(std::move(transportCosts), std::move(activityCosts)) {}

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
  using EvaluationContext = JobInsertionEvaluator::EvaluationContext;

  /// Analyzes tour trying to find best insertion index.
  InsertionResult analyze(const models::problem::Job& job,
                          const models::problem::Service& service,
                          const InsertionRouteContext& ctx,
                          const InsertionConstraint& constraint,
                          const InsertionProgress& progress) const {
    using namespace ranges;
    using namespace vrp::models;

    auto activity = models::solution::build_activity{}  //
                      .job(job)                         //
                      .shared();

    // calculate additional costs on route level.
    auto routeCosts = constraint.soft(ctx, job) + vehicleCosts(ctx);

    // form route legs from a new route view.
    auto [start, end] = waypoints(*ctx.actor, ctx.departure);
    auto tour = view::concat(view::single(start), ctx.route.first->tour.activities(), view::single(end));
    auto legs = view::zip(tour | view::sliding(2), view::iota(static_cast<size_t>(0)));
    auto evalCtx = EvaluationContext::empty(progress.bestCost, ctx.departure);

    // 1. analyze route legs
    auto result = ranges::accumulate(legs, evalCtx, [&](const auto& out, const auto& view) {
      if (out.isStopped) return out;

      auto [items, index] = view;
      auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
      auto actCtx = InsertionActivityContext{index, out.departure, prev, activity, next};

      // 2. analyze service details
      return ranges::accumulate(view::all(service.details), out, [&](const auto& in1, const auto& detail) {
        if (in1.isStopped) return in1;

        // TODO check whether tw is empty
        // 3. analyze detail time windows
        return ranges::accumulate(view::all(detail.times), in1, [&](const auto& in2, const auto& time) {
          if (in2.isStopped) return in2;

          activity->detail.time = time;
          activity->detail.duration = detail.duration;

          auto locations = detail.location.has_value()
            ? static_cast<any_view<common::Location>>(view::single(detail.location.value()))
            : view::concat(view::single(actCtx.prev->detail.location), view::single(actCtx.next->detail.location));

          // 4. analyze possible locations
          return ranges::accumulate(view::all(locations), in2, [&](const auto& in3, const auto& location) {
            if (in3.isStopped) return in3;

            activity->detail.location = location;

            // check hard activity constraint
            auto status = constraint.hard(ctx, actCtx);
            if (status.has_value()) return EvaluationContext::fail(status.value(), in3);

            // calculate all costs on activity level
            auto actCosts = constraint.soft(ctx, actCtx) + activityCosts(ctx, actCtx, progress);
            auto totalCosts = routeCosts + actCosts;

            // calculate end time (departure) for the next leg
            auto endTime = in3.departure + departure(*ctx.actor, *actCtx.prev, *actCtx.next, evalCtx.departure);

            return totalCosts < in3.cost
              ? EvaluationContext::success(actCtx.index, totalCosts, endTime, {location, detail.duration, time})
              : EvaluationContext::skip(endTime, in3);
          });
        });
      });
    });

    activity->detail = result.detail;

    return result.isSuccess() ? success(result, ctx, activity) : failure(result);
  }
};

}  // namespace vrp::algorithms::construction
