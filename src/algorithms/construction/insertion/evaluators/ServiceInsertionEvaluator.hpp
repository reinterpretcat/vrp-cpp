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
    auto activity = models::solution::build_activity{}            //
                      .withJob(models::problem::as_job(service))  //
                      .shared();

    // check hard constraints on route level.
    auto error = constraint_->hard(ctx, ranges::view::single(*activity));
    if (error.has_value()) { return {ranges::emplaced_index<1>, InsertionFailure{error.value()}}; }

    // calculate additional costs on route level.
    auto additionalCosts = constraint_->soft(ctx, ranges::view::single(*activity)) + extraCosts(ctx);

    auto result = analyze(activity, *service, ctx, progress, additionalCosts);

    return {ranges::emplaced_index<1>, InsertionFailure{}};
  }

private:
  using Activity = models::solution::Tour::Activity;
  using EvaluationContext = JobInsertionEvaluator::EvaluationContext;

  /// Analyzes tour trying to find best insertion index.
  InsertionResult analyze(models::solution::Tour::Activity& activity,
                          const models::problem::Service& service,
                          const InsertionRouteContext& routeCtx,
                          const InsertionProgress& progress,
                          models::common::Cost additionalCosts) const {
    using namespace ranges;
    using namespace vrp::models;

    // form route legs from a new route view.
    auto [start, end] = waypoints(routeCtx);
    auto tour = view::concat(view::single(start), routeCtx.route->tour.activities(), view::single(end));
    auto legs = view::zip(tour | view::sliding(2), view::iota(0));
    auto evalCtx = EvaluationContext::make_one(0, additionalCosts, routeCtx.time, 0, {});

    // 1. analyze route legs
    auto result = ranges::accumulate(legs, evalCtx, [&](const auto& outer, const auto& view) {
      if (outer.isInvalid()) return outer;

      // TODO recalculate departure
      auto [items, index] = view;
      auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
      auto actCtx = InsertionActivityContext{index, prev, activity, next};

      // 2. analyze service details
      return ranges::accumulate(view::all(service.details), outer, [&](const auto& inner1, const auto& detail) {
        if (inner1.isInvalid()) return inner1;

        // 3. analyze detail time windows
        return ranges::accumulate(view::all(detail.times), inner1, [&](const auto& inner2, const auto& time) {
          if (inner2.isInvalid()) return inner2;

          activity->time = time;
          activity->duration = detail.duration;

          auto locations = detail.location.has_value()
            ? static_cast<any_view<common::Location>>(view::single(detail.location.value()))
            : view::concat(view::single(actCtx.prev->location), view::single(actCtx.next->location));

          // 4. analyze possible locations
          return ranges::accumulate(view::all(locations), inner2, [&](const auto& inner3, const auto& location) {
            if (inner3.isInvalid()) return inner3;

            activity->location = location;

            // check hard activity constraint
            auto status = constraint_->hard(routeCtx, actCtx);
            if (status.has_value())
              return std::get<0>(status.value()) ? EvaluationContext::make_invalid(std::get<1>(status.value()))
                                                 : inner3;

            // calculate all costs on activity level
            auto activityCosts = constraint_->soft(routeCtx, actCtx) + extraCosts(routeCtx, actCtx, progress);
            auto totalCosts = additionalCosts + activityCosts;

            return totalCosts < progress.bestCost
              // TODO is departure value valid here?
              ? EvaluationContext::make_one(actCtx.index, totalCosts, inner3.departure, location, time)
              : inner3;
          });
        });
      });
    });

    // TODO analyze result
    return {};
  }

  /// Creates start/end stops of vehicle.
  std::pair<Activity, Activity> waypoints(const InsertionRouteContext& ctx) const {
    using namespace vrp::utils;
    using namespace vrp::models;

    // create start/end for new vehicle
    auto start = solution::build_activity{}
                   .withLocation(ctx.actor->vehicle->start)                                                        //
                   .withSchedule({ctx.actor->vehicle->time.start, std::numeric_limits<common::Timestamp>::max()})  //
                   .shared();
    auto end = solution::build_activity{}
                 .withLocation(ctx.actor->vehicle->end.value_or(ctx.actor->vehicle->start))  //
                 .withSchedule({0, ctx.actor->vehicle->time.end})                            //
                 .shared();

    return {start, end};
  }

  std::shared_ptr<const InsertionConstraint> constraint_;
};

}  // namespace vrp::algorithms::construction
