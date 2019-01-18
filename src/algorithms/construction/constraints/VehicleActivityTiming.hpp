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
  void accept(models::solution::Route& route, InsertionRouteState& state) const override {
    using namespace ranges;

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
      auto init = std::pair{pair.second.first, pair.second.second};

      ranges::accumulate(view::reverse(route.tour.activities()), init, [&](const auto& acc, const auto& act) {
        const auto& [endTime, prevLoc] = acc;

        auto potentialLatest = endTime -
          transport_->duration(actor.vehicle->profile, act->detail.location, prevLoc, endTime) -
          activity_->duration(actor, *act, endTime);
        auto latestArrivalTime = std::min(act->detail.time.end, potentialLatest);

        if (latestArrivalTime < act->detail.time.start) state.put<bool>(stateKey, true);

        state.put<models::common::Timestamp>(stateKey, *act, latestArrivalTime);

        return std::make_pair(latestArrivalTime, act->detail.location);
      });
    });
  }

  /// Checks whether proposed vehicle can be used within route without violating time windows.
  HardRouteConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                   const HardRouteConstraint::Job&) const override {
    return routeCtx.route.second->get<bool>(actorSharedKey(StateKey, *routeCtx.actor)).value_or(false)
      ? HardRouteConstraint::Result{code_}
      : HardRouteConstraint::Result{};
  }

  /// Checks whether proposed activity insertion doesn't violate time windows.
  HardActivityConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                      const InsertionActivityContext& actCtx) const override {
    using namespace vrp::models;
    using namespace vrp::models::common;

    const auto& actor = *routeCtx.actor;
    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = *actCtx.next;

    auto latestArrival = actor.detail.time.end;
    auto nextActLocation = next.type == solution::Activity::Type::End ? actor.detail.end.value_or(next.detail.location)
                                                                      : next.detail.location;
    auto latestArrTimeAtNextAct = next.type == solution::Activity::Type::End
      ? actor.detail.time.end
      : routeCtx.route.second->get<Timestamp>(actorSharedKey(StateKey, actor), next).value_or(next.detail.time.end);

    if (latestArrival < prev.detail.time.start || latestArrival < target.detail.time.start ||
        latestArrival < next.detail.time.start)
      return fail(code_);

    if (target.detail.time.end < prev.detail.time.start) return fail(code_);

    auto arrTimeAtNext = actCtx.departure +
      transport_->duration(actor.vehicle->profile, prev.detail.location, nextActLocation, actCtx.departure);
    if (arrTimeAtNext > latestArrTimeAtNextAct) return fail(code_);

    if (target.detail.time.start > next.detail.time.end) return stop(code_);

    auto arrTimeAtTargetAct = actCtx.departure +
      transport_->duration(actor.vehicle->profile, prev.detail.location, target.detail.location, actCtx.departure);

    auto endTimeAtNewAct =
      std::max(arrTimeAtTargetAct, target.detail.time.start) + activity_->duration(actor, target, arrTimeAtTargetAct);

    auto latestArrTimeAtNewAct = std::min(
      target.detail.time.end,
      latestArrTimeAtNextAct -
        transport_->duration(actor.vehicle->profile, target.detail.location, nextActLocation, latestArrTimeAtNextAct) +
        activity_->duration(actor, target, arrTimeAtTargetAct));

    if (arrTimeAtTargetAct > latestArrTimeAtNewAct) return stop(code_);

    if (next.type == solution::Activity::Type::End && !actor.detail.end.has_value()) return success();

    auto arrTimeAtNextAct = endTimeAtNewAct +
      transport_->duration(actor.vehicle->profile, target.detail.location, nextActLocation, endTimeAtNewAct);

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
    const auto& route = routeCtx.route.first;

    auto [tpCostLeft, actCostLeft, depTimeLeft] = analyze(*routeCtx.actor, prev, target, actCtx.departure);

    auto [tpCostRight, actCostRight, depTimeRight] = analyze(*routeCtx.actor, target, next, depTimeLeft);

    auto newCosts = tpCostLeft + tpCostRight + /* progress.completeness * */ (actCostLeft + actCostRight);

    if (routeCtx.route.first->tour.empty()) return newCosts;

    auto [tpCostOld, actCostOld, depTimeOld] =
      analyze(*route->actor,
              prev.type == Activity::Type::Start ? *route->start : prev,
              next.type == Activity::Type::End ? *route->end : next,
              prev.type == Activity::Type::Start ? route->start->schedule.departure : prev.schedule.departure);

    auto oldCosts = tpCostOld + /*progress.completeness * */ actCostOld;

    return newCosts - oldCosts;

    return {};
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
