#pragma once

#include "algorithms/construction/insertion/InsertionActivityContext.hpp"
#include "algorithms/construction/insertion/InsertionProgress.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"

#include <algorithm>
#include <limits>

namespace vrp::algorithms::construction {

struct JobInsertionEvaluator {
  explicit JobInsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                                 std::shared_ptr<const models::costs::ActivityCosts> activityCosts) :
    transportCosts_(std::move(transportCosts)),
    activityCosts_(std::move(activityCosts)) {}

  virtual ~JobInsertionEvaluator() = default;

protected:
  /// Specifies evaluation context.
  struct EvaluationContext final {
    /// Insertion index.
    size_t index = 0;
    /// Violation code.
    int code = -1;
    /// Vehicle departure time.
    models::common::Timestamp departure = 0;
    /// Activity location.
    models::common::Location location = 0;
    /// Activity arrival and departure limits.
    models::common::TimeWindow tw = {0, 0};
    /// Checks whether context is invalidaded.
    bool isInvalid() const { return code >= 0; }
    /// Creates invalidated context.
    static EvaluationContext make_invalid(int code) { return EvaluationContext{0, code, 0, 0, {0, 0}}; }
  };

  /// Estimates extra costs on route level.
  models::common::Cost extraCosts(const InsertionRouteContext& ctx) const {
    models::common::Cost deltaFirst = 0.0;
    models::common::Cost deltaLast = 0.0;

    if (!ctx.route->tour.empty()) {
      double firstCostNew = transportCosts_->cost(*ctx.actor,
                                                  ctx.actor->vehicle->start,          //
                                                  ctx.route->tour.first()->location,  //
                                                  ctx.time);
      double firstCostOld = transportCosts_->cost(ctx.route->actor,
                                                  ctx.route->start->location,         //
                                                  ctx.route->tour.first()->location,  //
                                                  ctx.route->start->schedule.departure);

      deltaFirst = firstCostNew - firstCostOld;

      if (ctx.actor->vehicle->end.has_value()) {
        auto last = ctx.route->tour.last();
        auto lastDepartureOld = last->schedule.departure;
        auto lastDepartureNew = std::max(models::common::Timestamp{0},  //
                                         lastDepartureOld + (ctx.time - ctx.route->start->schedule.departure));

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

  /// Estimates extra costs on route level.
  models::common::Cost extraCosts(const InsertionRouteContext& routeCtx,
                                  const InsertionActivityContext& actCtx,
                                  const InsertionProgress& progress) const {
    using namespace vrp::models::common;
    using namespace vrp::models::solution;

    // TODO extract to function which returns pair

    auto tp_costs_prevAct_newAct =
      transportCosts_->cost(*routeCtx.actor, actCtx.prev->location, actCtx.next->location, actCtx.time);
    auto tp_time_prevAct_newAct =
      transportCosts_->duration(*routeCtx.actor, actCtx.prev->location, actCtx.next->location, actCtx.time);

    auto newAct_arrTime = actCtx.time + tp_time_prevAct_newAct;
    auto newAct_endTime = std::max(newAct_arrTime, actCtx.target->time.start) +
      activityCosts_->duration(*routeCtx.actor, *actCtx.target, newAct_arrTime);

    auto act_costs_newAct = activityCosts_->cost(*routeCtx.actor, *actCtx.target, newAct_arrTime);

    // next location is the end which is not depot
    if (actCtx.next->type == Activity::Type::End && routeCtx.actor->vehicle->end.has_value())
      return tp_costs_prevAct_newAct + progress.completeness * act_costs_newAct;

    auto tp_costs_newAct_nextAct =
      transportCosts_->cost(*routeCtx.actor, actCtx.target->location, actCtx.next->location, newAct_endTime);
    auto tp_time_newAct_nextAct =
      transportCosts_->duration(*routeCtx.actor, actCtx.target->location, actCtx.next->location, newAct_endTime);
    auto nextAct_arrTime = newAct_endTime + tp_time_newAct_nextAct;
    auto endTime_nextAct_new = std::max(nextAct_arrTime, actCtx.next->time.start) +
      activityCosts_->duration(*routeCtx.actor, *actCtx.next, newAct_arrTime);
    auto act_costs_nextAct = activityCosts_->cost(*routeCtx.actor, *actCtx.next, nextAct_arrTime);

    auto totalCosts = tp_costs_prevAct_newAct + tp_costs_newAct_nextAct +
      progress.completeness * (act_costs_newAct + act_costs_nextAct);

    auto oldCosts = 0.;
    if (routeCtx.route->tour.empty()) {
      oldCosts += transportCosts_->cost(*routeCtx.actor, actCtx.prev->location, actCtx.next->location, actCtx.time);
    } else {
      auto tp_costs_prevAct_nextAct = transportCosts_->cost(
        routeCtx.route->actor, actCtx.prev->location, actCtx.next->location, actCtx.prev->schedule.departure);
      auto arrTime_nextAct = actCtx.time +
        transportCosts_->duration(
          routeCtx.route->actor, actCtx.prev->location, actCtx.next->location, actCtx.prev->schedule.departure);
      auto endTime_nextAct_old = std::max(arrTime_nextAct, actCtx.next->time.start) +
        activityCosts_->duration(routeCtx.route->actor, *actCtx.next, arrTime_nextAct);
      auto actCost_nextAct = activityCosts_->cost(routeCtx.route->actor, *actCtx.next, arrTime_nextAct);

      auto endTimeDelay_nextAct = std::max(Timestamp{0}, endTime_nextAct_new - endTime_nextAct_old);

      auto futureWaiting = routeCtx.state->get<Timestamp>(InsertionRouteState::FutureWaiting, *actCtx.next).value_or(0);
      auto waitingTime_savings_timeUnit = std::min(futureWaiting, endTimeDelay_nextAct);
      auto waitingTime_savings = waitingTime_savings_timeUnit * routeCtx.route->actor.vehicle->costs.perWaitingTime;
      oldCosts += progress.completeness * waitingTime_savings;
      oldCosts += tp_costs_prevAct_nextAct + progress.completeness * actCost_nextAct;
    }

    return totalCosts - oldCosts;
  }

private:
  std::shared_ptr<const models::costs::TransportCosts> transportCosts_;
  std::shared_ptr<const models::costs::ActivityCosts> activityCosts_;
};
}
