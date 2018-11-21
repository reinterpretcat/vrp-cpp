#pragma once

#include "algorithms/construction/insertion/InsertionActivityContext.hpp"
#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "algorithms/construction/insertion/evaluators/JobInsertionEvaluator.hpp"
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
                            std::shared_ptr<const models::costs::ActivityCosts> activityCosts,
                            std::shared_ptr<const InsertionConstraint> constraint) :
    JobInsertionEvaluator(std::move(transportCosts), std::move(activityCosts)),
    constraint_(std::move(constraint)) {}

  /// Evaluates service insertion possibility.
  InsertionResult evaluate(const std::shared_ptr<const models::problem::Service>& service,
                           const InsertionRouteContext& ctx,
                           const InsertionProgress& progress) const {
    auto activity = models::solution::build_activity{}        //
                      .job(models::problem::as_job(service))  //
                      .shared();

    // check hard constraints on route level.
    auto error = constraint_->hard(ctx, ranges::view::single(*activity));
    if (error.has_value()) return {ranges::emplaced_index<1>, InsertionFailure{error.value()}};

    return analyze(activity, *service, ctx, progress);
  }

private:
  using Activity = models::solution::Tour::Activity;
  using EvaluationContext = JobInsertionEvaluator::EvaluationContext;

  /// Analyzes tour trying to find best insertion index.
  InsertionResult analyze(models::solution::Tour::Activity& activity,
                          const models::problem::Service& service,
                          const InsertionRouteContext& routeCtx,
                          const InsertionProgress& progress) const {
    using namespace ranges;
    using namespace vrp::models;

    // calculate additional costs on route level.
    auto routeCosts = constraint_->soft(routeCtx, view::single(*activity)) + vehicleCosts(routeCtx);

    // form route legs from a new route view.
    auto [start, end] = waypoints(routeCtx);
    auto tour = view::concat(view::single(start), routeCtx.route->tour.activities(), view::single(end));
    auto legs = view::zip(tour | view::sliding(2), view::iota(static_cast<size_t>(0)));
    auto evalCtx = EvaluationContext::make_one(0, progress.bestCost, routeCtx.departure, 0, 0, {});

    // 1. analyze route legs
    auto result = ranges::accumulate(legs, evalCtx, [&](const auto& out, const auto& view) {
      if (out.isInvalid()) return out;

      // TODO recalculate departure
      auto [items, index] = view;
      auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
      auto actCtx = InsertionActivityContext{index, out.departure, prev, activity, next};

      // 2. analyze service details
      return ranges::accumulate(view::all(service.details), out, [&](const auto& in1, const auto& detail) {
        if (in1.isInvalid()) return in1;

        // TODO check whether tw is empty
        // 3. analyze detail time windows
        return ranges::accumulate(view::all(detail.times), in1, [&](const auto& in2, const auto& time) {
          if (in2.isInvalid()) return in2;

          activity->time = time;
          activity->duration = detail.duration;

          auto locations = detail.location.has_value()
            ? static_cast<any_view<common::Location>>(view::single(detail.location.value()))
            : view::concat(view::single(actCtx.prev->location), view::single(actCtx.next->location));

          // 4. analyze possible locations
          return ranges::accumulate(view::all(locations), in2, [&](const auto& in3, const auto& location) {
            if (in3.isInvalid()) return in3;

            activity->location = location;

            // check hard activity constraint
            auto status = constraint_->hard(routeCtx, actCtx);
            if (status.has_value())
              return std::get<0>(status.value()) ? EvaluationContext::make_invalid(std::get<1>(status.value())) : in3;

            // calculate all costs on activity level
            auto actCosts = constraint_->soft(routeCtx, actCtx) + activityCosts(routeCtx, actCtx, progress);
            auto totalCosts = routeCosts + actCosts;

            // calculate end time (departure) for the next leg
            auto endTime = in3.departure + departure(*routeCtx.actor, *actCtx.prev, *actCtx.next, evalCtx.departure);

            return totalCosts < in3.bestCost
              ? EvaluationContext::make_one(actCtx.index, totalCosts, endTime, detail.duration, location, time)
              : EvaluationContext::make_one(in3.index, in3.bestCost, endTime, in3.duration, in3.location, in3.tw);
          });
        });
      });
    });

    activity->location = result.location;
    activity->duration = result.duration;
    activity->time = result.tw;

    return result.isInvalid()
      ? InsertionResult{ranges::emplaced_index<1>, InsertionFailure{result.code}}
      : InsertionResult{ranges::emplaced_index<0>,
                        InsertionSuccess{result.index, activity, routeCtx.actor, routeCtx.departure}};
  }

  /// Creates start/end stops of vehicle.
  std::pair<Activity, Activity> waypoints(const InsertionRouteContext& ctx) const {
    using namespace vrp::utils;
    using namespace vrp::models;

    // create start/end for new vehicle
    auto start = solution::build_activity{}
                   .type(solution::Activity::Type::Start)
                   .time({ctx.actor->time.start, std::numeric_limits<common::Timestamp>::max()})
                   .location(ctx.actor->start)                        //
                   .schedule({ctx.actor->time.start, ctx.departure})  //
                   .shared();
    auto end = solution::build_activity{}
                 .type(solution::Activity::Type::End)
                 .time({0, ctx.actor->time.end})
                 .location(ctx.actor->end.value_or(ctx.actor->start))  //
                 .schedule({0, ctx.actor->time.end})                   //
                 .shared();

    return {start, end};
  }

  std::shared_ptr<const InsertionConstraint> constraint_;
};

}  // namespace vrp::algorithms::construction
