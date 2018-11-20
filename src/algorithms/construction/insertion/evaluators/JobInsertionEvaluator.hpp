#pragma once

#include "algorithms/construction/insertion/InsertionActivityContext.hpp"
#include "algorithms/construction/insertion/InsertionProgress.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

namespace vrp::algorithms::construction {

struct JobInsertionEvaluator {
  JobInsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                        std::shared_ptr<const models::costs::ActivityCosts> activityCosts) :
    transportCosts_(std::move(transportCosts)),
    activityCosts_(std::move(activityCosts)) {}

  virtual ~JobInsertionEvaluator() = default;

protected:
  /// Specifies evaluation context.
  struct EvaluationContext final {
    /// Violation code.
    int code = -1;
    /// Insertion index.
    size_t index = 0;
    /// Best cost.
    models::common::Cost bestCost = std::numeric_limits<models::common::Cost>::max();
    /// Activity departure time.
    models::common::Timestamp departure = 0;
    /// Activity departure time.
    models::common::Duration duration = 0;
    /// Activity location.
    models::common::Location location = 0;
    /// Activity arrival and departure limits.
    models::common::TimeWindow tw = {0, 0};

    /// Checks whether context is invalidated.
    bool isInvalid() const { return code >= 0; }

    /// Creates invalidated context.
    static EvaluationContext make_invalid(int code) {
      return EvaluationContext{code, 0, std::numeric_limits<models::common::Cost>::max(), 0, 0, 0, {0, 0}};
    }

    /// Creates new context.
    static EvaluationContext make_one(size_t index,
                                      const models::common::Cost& bestCost,
                                      const models::common::Timestamp& departure,
                                      const models::common::Duration& duration,
                                      const models::common::Location& location,
                                      const models::common::TimeWindow& tw) {
      return {-1, index, bestCost, departure, duration, location, tw};
    }
  };

  /// Calculates vehicle specific costs.
  models::common::Cost vehicleCosts(const InsertionRouteContext& ctx) const {
    models::common::Cost deltaFirst = 0.0;
    models::common::Cost deltaLast = 0.0;

    if (!ctx.route->tour.empty()) {
      double firstCostNew = transportCosts_->cost(*ctx.actor,
                                                  ctx.actor->vehicle->start,          //
                                                  ctx.route->tour.first()->location,  //
                                                  ctx.departure);
      double firstCostOld = transportCosts_->cost(ctx.route->actor,
                                                  ctx.route->start->location,         //
                                                  ctx.route->tour.first()->location,  //
                                                  ctx.route->start->schedule.departure);

      deltaFirst = firstCostNew - firstCostOld;

      if (ctx.actor->vehicle->end.has_value()) {
        auto last = ctx.route->tour.last();
        auto lastDepartureOld = last->schedule.departure;
        auto lastDepartureNew = std::max(models::common::Timestamp{0},  //
                                         lastDepartureOld + (ctx.departure - ctx.route->start->schedule.departure));

        auto lastCostNew = transportCosts_->cost(*ctx.actor,
                                                 last->location,                   //
                                                 ctx.actor->vehicle->end.value(),  //
                                                 lastDepartureNew);
        auto lastCostOld = transportCosts_->cost(ctx.route->actor,
                                                 last->location,            //
                                                 ctx.route->end->location,  //
                                                 lastDepartureNew);

        deltaLast = lastCostNew - lastCostOld;
      }
    }

    return deltaFirst + deltaLast;
  }

  /// Calculates activity insertion costs locally, i.e. by comparing extra costs of
  /// insertion the new activity k between activity i and j.
  /// Additional costs are then basically calculated as delta c = c_ik + c_kj - c_ij.
  models::common::Cost activityCosts(const InsertionRouteContext& routeCtx,
                                     const InsertionActivityContext& actCtx,
                                     const InsertionProgress& progress) const {
    using namespace vrp::models::common;
    using namespace vrp::models::solution;

    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = *actCtx.next;

    auto [tpCostLeft, actCostLeft, depTimeLeft] = analyzeLeg(*routeCtx.actor, prev, target, actCtx.departure);

    // next location is the end which is not depot
    if (actCtx.next->type == Activity::Type::End && routeCtx.actor->vehicle->end.has_value())
      return tpCostLeft + progress.completeness * actCostLeft;

    auto [tpCostRight, actCostRight, depTimeRight] = analyzeLeg(*routeCtx.actor, target, next, depTimeLeft);

    auto totalCosts = tpCostLeft + tpCostRight + progress.completeness * (actCostLeft + actCostRight);
    auto oldCosts = 0.;

    if (routeCtx.route->tour.empty()) {
      oldCosts += transportCosts_->cost(*routeCtx.actor, prev.location, next.location, actCtx.departure);
    } else {
      auto [tpCostOld, actCostOld, depTimeOld] = analyzeLeg(routeCtx.route->actor, prev, next, prev.schedule.departure);

      auto delayTime = depTimeRight > depTimeOld ? depTimeRight - depTimeOld : 0;
      auto futureWaiting = Timestamp{0};  // routeCtx.state->get<Timestamp>("FutureWaiting", next).value_or(0);
      auto timeCostSavings = std::min(futureWaiting, delayTime) * routeCtx.route->actor.vehicle->costs.perWaitingTime;

      oldCosts += tpCostOld + progress.completeness * (actCostOld + timeCostSavings);
    }

    return totalCosts - oldCosts;
  }

  /// Returns departure time from end activity taking into account time: departure time from start activity.
  models::common::Duration departure(const models::problem::Actor& actor,
                                     const models::solution::Activity& start,
                                     const models::solution::Activity& end,
                                     const models::common::Timestamp& depTime) const {
    auto arrival = depTime + transportCosts_->duration(actor, start.location, end.location, depTime);
    return std::max(arrival, end.time.start) + activityCosts_->duration(actor, end, arrival);
  }

private:
  using Cost = models::common::Cost;
  using Timestamp = models::common::Timestamp;

  std::tuple<Cost, Cost, Timestamp> analyzeLeg(const models::problem::Actor& actor,
                                               const models::solution::Activity& start,
                                               const models::solution::Activity& end,
                                               models::common::Timestamp time) const {
    auto arrival = time + transportCosts_->duration(actor, start.location, end.location, time);
    auto departure = std::max(arrival, end.time.start) + activityCosts_->duration(actor, end, arrival);

    auto transportCost = transportCosts_->cost(actor, start.location, end.location, time);
    auto activityCost = activityCosts_->cost(actor, end, arrival);

    return std::make_tuple(transportCost, activityCost, departure);
  }

  std::shared_ptr<const models::costs::TransportCosts> transportCosts_;
  std::shared_ptr<const models::costs::ActivityCosts> activityCosts_;
};
}
