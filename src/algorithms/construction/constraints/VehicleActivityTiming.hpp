#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Constraints.hpp"
#include "algorithms/construction/extensions/Fleets.hpp"
#include "algorithms/construction/extensions/States.hpp"
#include "models/Problem.hpp"
#include "models/common/Location.hpp"
#include "models/common/Timestamp.hpp"
#include "models/costs/ActivityCosts.hpp"
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
  inline static const std::string StateKey = "op_time";
  inline static const std::string WaitingKey = "fw_time";

  VehicleActivityTiming(const std::shared_ptr<const models::problem::Fleet>& fleet,
                        const std::shared_ptr<const models::costs::TransportCosts>& transport,
                        const std::shared_ptr<const models::costs::ActivityCosts>& activity,
                        int code = 1) :
    code_(code),
    keys_(),
    transport_(transport),
    activity_(activity) {
    ranges::for_each(empty_actors(*fleet), [&](const auto& a) {
      auto key = actorSharedKey(StateKey, a);
      if (keys_.find(key) == keys_.end()) {
        keys_[key] = std::make_pair(a.detail.time.end, a.detail.end.value_or(a.detail.start));
      }
    });
    assert(!keys_.empty());
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
      view::concat(view::single(route.start), route.tour.activities(), view::single(route.end)),
      std::pair{route.start->detail.location, route.start->schedule.departure},
      [&](const auto& acc, auto& a) {
        const auto& [loc, dep] = acc;

        a->schedule.arrival = dep + transport_->duration(actor.vehicle->profile, loc, a->detail.location, dep);
        a->schedule.departure =
          std::max(a->schedule.arrival, a->detail.time.start) + activity_->duration(actor, *a, a->schedule.arrival);

        return std::pair{a->detail.location, a->schedule.departure};
      });

    // update latest possible arrivals for each unique actor type
    ranges::for_each(view::all(keys_), [&](const auto& pair) {
      const auto& stateKey = pair.first;
      auto init = std::tuple{pair.second.first, pair.second.second, Timestamp{0}};

      ranges::accumulate(view::reverse(context.route->tour.activities()), init, [&](const auto& acc, const auto& act) {
        const auto& [endTime, prevLoc, waiting] = acc;

        auto potentialLatest = endTime -
          transport_->duration(actor.vehicle->profile, act->detail.location, prevLoc, endTime) -
          activity_->duration(actor, *act, endTime);
        auto latestArrivalTime = std::min(act->detail.time.end, potentialLatest);

        auto futureWaiting = waiting + std::max(act->detail.time.start - act->schedule.arrival, Timestamp{0});

        if (latestArrivalTime < act->detail.time.start) context.state->put<bool>(stateKey, true);

        context.state->put<Timestamp>(stateKey, *act, latestArrivalTime);
        context.state->put<Timestamp>(WaitingKey, *act, futureWaiting);

        return std::make_tuple(latestArrivalTime, act->detail.location, futureWaiting);
      });
    });
  }

  /// Checks whether proposed vehicle can be used within route without violating time windows.
  HardRouteConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                   const HardRouteConstraint::Job&) const override {
    return routeCtx.state->get<bool>(actorSharedKey(StateKey, *routeCtx.route->actor)).value_or(false)
      ? HardRouteConstraint::Result{code_}
      : HardRouteConstraint::Result{};
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
    auto nextActLocation = next.type == solution::Activity::Type::End ? actor.detail.end.value_or(next.detail.location)
                                                                      : next.detail.location;
    auto latestArrTimeAtNextAct = next.type == solution::Activity::Type::End
      ? actor.detail.time.end
      : routeCtx.state->get<Timestamp>(actorSharedKey(StateKey, actor), next).value_or(next.detail.time.end);

    if (latestArrival < prev.detail.time.start || latestArrival < target.detail.time.start ||
        latestArrival < next.detail.time.start)
      return fail(code_);

    if (target.detail.time.end < prev.detail.time.start) return fail(code_);

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

    if (next.type == solution::Activity::Type::End && !actor.detail.end.has_value()) return success();

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

    if (route.tour.empty()) return newCosts;

    auto [tpCostOld, actCostOld, depTimeOld] =
      analyze(actor,
              prev.type == Activity::Type::Start ? *route.start : prev,
              next.type == Activity::Type::End ? *route.end : next,
              prev.type == Activity::Type::Start ? route.start->schedule.departure : prev.schedule.departure);

    auto waitingTime = routeCtx.state->get<Timestamp>(actorSharedKey(WaitingKey, actor), next).value_or(Timestamp{0});
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
  std::unordered_map<std::string, std::pair<models::common::Timestamp, models::common::Location>> keys_;
  std::shared_ptr<const models::costs::TransportCosts> transport_;
  std::shared_ptr<const models::costs::ActivityCosts> activity_;
};
}
