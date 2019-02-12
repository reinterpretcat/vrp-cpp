#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Constraints.hpp"
#include "models/Problem.hpp"
#include "models/common/Location.hpp"
#include "models/common/Timestamp.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/extensions/solution/Comparators.hpp"
#include "models/problem/Fleet.hpp"

#include <algorithm>
#include <optional>
#include <range/v3/all.hpp>
#include <string>
#include <utility>

namespace vrp::algorithms::construction {

/// Checks whether vehicle can serve activity taking into account their time windows.
struct VehicleActivityTiming final
  : public HardRouteConstraint
  , public HardActivityConstraint
  , public SoftRouteConstraint
  , public SoftActivityConstraint {
  constexpr static int BaseKey = 0;
  constexpr static int LatestArrivalKey = BaseKey + 0;
  constexpr static int WaitingKey = BaseKey + 1;

  VehicleActivityTiming(const std::shared_ptr<const models::problem::Fleet>& fleet,
                        const std::shared_ptr<const models::costs::TransportCosts>& transport,
                        const std::shared_ptr<const models::costs::ActivityCosts>& activity,
                        int code = 1) :
    code_(code),
    transport_(transport),
    activity_(activity) {}

  /// Returns used state keys.
  ranges::any_view<int> stateKeys() const override {
    using namespace ranges;
    return view::concat(view::single(LatestArrivalKey), view::single(WaitingKey));
  }

  /// Accept route and updates its insertion state.
  void accept(InsertionRouteContext& context) const override {
    using namespace ranges;
    using namespace models::common;

    const auto& route = *context.route;
    const auto& state = *context.state;
    const auto& actor = *route.actor;

    // update each activity schedule
    ranges::accumulate(  //
      route.tour.activities(),
      std::pair{route.tour.start()->detail.location, route.tour.start()->schedule.departure},
      [&](const auto& acc, auto& a) {
        const auto& [loc, dep] = acc;

        a->schedule.arrival = dep + transport_->duration(actor.vehicle->profile, loc, a->detail.location, dep);

        a->schedule.departure =
          std::max(a->schedule.arrival, a->detail.time.start) + activity_->duration(actor, *a, a->schedule.arrival);

        return std::pair{a->detail.location, a->schedule.departure};
      });

    // update latest arrival and waiting states of non-terminate (jobs) activities
    auto init = std::tuple{actor.detail.time.end, actor.detail.end.value_or(actor.detail.start), Timestamp{0}};
    ranges::accumulate(view::reverse(context.route->tour.activities()), init, [&](const auto& acc, const auto& act) {
      if (!act->service.has_value()) return acc;

      const auto& [endTime, prevLoc, waiting] = acc;

      auto potentialLatest = endTime -
        transport_->duration(actor.vehicle->profile, act->detail.location, prevLoc, endTime) -
        activity_->duration(actor, *act, endTime);
      auto latestArrivalTime = std::min(act->detail.time.end, potentialLatest);

      auto futureWaiting = waiting + std::max(act->detail.time.start - act->schedule.arrival, Timestamp{0});

      context.state->put<Timestamp>(LatestArrivalKey, act, latestArrivalTime);
      context.state->put<Timestamp>(WaitingKey, act, futureWaiting);

      return std::make_tuple(latestArrivalTime, act->detail.location, futureWaiting);
    });
  }

  /// Checks whether proposed vehicle can be used within route without violating time windows.
  HardRouteConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                   const HardRouteConstraint::Job&) const override {
    // TODO check that job's and actor's TWs have intersection
    return HardRouteConstraint::Result{};
  }

  /// Checks whether proposed activity insertion doesn't violate time windows.
  HardActivityConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                      const InsertionActivityContext& actCtx) const override {
    using namespace vrp::models;
    using namespace vrp::models::common;

    const auto& actor = *routeCtx.route->actor;
    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = *actCtx.next;

    const auto departure = prev.schedule.departure;
    const auto& profile = actor.vehicle->profile;

    auto latestArrival = actor.detail.time.end;
    if (latestArrival < prev.detail.time.start || latestArrival < target.detail.time.start ||
        latestArrival < next.detail.time.start)
      return fail(code_);

    if (target.detail.time.end < prev.detail.time.start) return fail(code_);

    auto nextActLocation = next.service.has_value()  //
      ? next.detail.location
      : actor.detail.end.value_or(next.detail.location);
    auto latestArrTimeAtNextAct = next.service.has_value()
      ? routeCtx.state->get<Timestamp>(LatestArrivalKey, actCtx.next).value_or(next.detail.time.end)
      : actor.detail.time.end;

    auto arrTimeAtNext = departure + transport_->duration(profile, prev.detail.location, nextActLocation, departure);
    if (arrTimeAtNext > latestArrTimeAtNextAct) return fail(code_);

    if (target.detail.time.start > next.detail.time.end) return stop(code_);

    auto arrTimeAtTargetAct =
      departure + transport_->duration(profile, prev.detail.location, target.detail.location, departure);

    auto endTimeAtNewAct =
      std::max(arrTimeAtTargetAct, target.detail.time.start) + activity_->duration(actor, target, arrTimeAtTargetAct);

    auto latestArrTimeAtNewAct =
      std::min(target.detail.time.end,
               latestArrTimeAtNextAct -
                 transport_->duration(profile, target.detail.location, nextActLocation, latestArrTimeAtNextAct) +
                 activity_->duration(actor, target, arrTimeAtTargetAct));

    if (arrTimeAtTargetAct > latestArrTimeAtNewAct) return stop(code_);

    if (!next.service.has_value() && !actor.detail.end.has_value()) return success();

    auto arrTimeAtNextAct =
      endTimeAtNewAct + transport_->duration(profile, target.detail.location, nextActLocation, endTimeAtNewAct);

    return arrTimeAtNextAct > latestArrTimeAtNextAct ? stop(code_) : success();
  }

  /// TODO
  models::common::Cost soft(const InsertionRouteContext& routeCtx, const SoftRouteConstraint::Job&) const override {
    return {};
  }

  /// Calculates activity insertion costs locally, i.e. by comparing extra costs of
  /// insertion the new activity k between activity i and j.
  /// Additional costs are then basically calculated as delta c = c_ik + c_kj - c_ij.
  models::common::Cost soft(const InsertionRouteContext& routeCtx,
                            const InsertionActivityContext& actCtx) const override {
    using namespace vrp::models::common;
    using namespace vrp::models::solution;

    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = *actCtx.next;

    const auto& route = *routeCtx.route;
    const auto& actor = *route.actor;

    auto [tpCostLeft, actCostLeft, depTimeLeft] = analyze(actor, prev, target, prev.schedule.departure);

    auto [tpCostRight, actCostRight, depTimeRight] = analyze(actor, target, next, depTimeLeft);

    auto newCosts = tpCostLeft + tpCostRight + /* progress.completeness * */ (actCostLeft + actCostRight);

    if (!route.tour.hasJobs()) return newCosts;

    auto [tpCostOld, actCostOld, depTimeOld] =
      analyze(actor,
              prev.service.has_value() ? prev : *route.tour.start(),
              next.service.has_value() ? next : *route.tour.end(),
              prev.service.has_value() ? prev.schedule.departure : route.tour.start()->schedule.departure);

    auto waitingTime = routeCtx.state->get<Timestamp>(WaitingKey, actCtx.next).value_or(Timestamp{0});

    double waitingCost =
      std::min(waitingTime, std::max(Timestamp{0}, depTimeRight - depTimeOld)) * actor.vehicle->costs.perWaitingTime;

    auto oldCosts = tpCostOld + /*progress.completeness * */ (actCostOld + waitingCost);

    return newCosts - oldCosts;
  }

private:
  using Cost = models::common::Cost;
  using Timestamp = models::common::Timestamp;

  /// Analyzes route leg.
  std::tuple<Cost, Cost, Timestamp> analyze(const models::solution::Actor& actor,
                                            const models::solution::Activity& start,
                                            const models::solution::Activity& end,
                                            models::common::Timestamp time) const {
    auto arrival =
      time + transport_->duration(actor.vehicle->profile, start.detail.location, end.detail.location, time);
    auto departure = std::max(arrival, end.detail.time.start) + activity_->duration(actor, end, arrival);

    auto transportCost = transport_->cost(actor, start.detail.location, end.detail.location, time);
    auto activityCost = activity_->cost(actor, end, arrival);

    return std::make_tuple(transportCost, activityCost, departure);
  }

  int code_;
  std::shared_ptr<const models::costs::TransportCosts> transport_;
  std::shared_ptr<const models::costs::ActivityCosts> activity_;
};
}
