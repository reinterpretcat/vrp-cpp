#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Constraints.hpp"
#include "algorithms/construction/extensions/Fleets.hpp"
#include "algorithms/construction/extensions/States.hpp"
#include "models/common/Location.hpp"
#include "models/common/Timestamp.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
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
  , public HardActivityConstraint {
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

        auto duration =
          std::max<std::int64_t>(0,
                                 transport_->duration(actor.vehicle->profile, act->detail.location, prevLoc, endTime) -
                                   activity_->duration(actor, *act, endTime));
        auto potentialLatest = endTime > duration ? endTime - duration : 0;

        auto latestArrivalTime = std::min(act->detail.time.end, potentialLatest);
        if (latestArrivalTime < act->detail.time.start || endTime < duration) state.put<bool>(stateKey, true);

        state.put<models::common::Timestamp>(stateKey, *act, latestArrivalTime);

        return std::make_pair(latestArrivalTime, act->detail.location);
      });
    });
  }

  /// Checks whether proposed vehicle can be used within route without violating time windows.
  HardRouteConstraint::Result check(const InsertionRouteContext& routeCtx,
                                    const HardRouteConstraint::Job&) const override {
    return routeCtx.route.second->get<bool>(actorSharedKey(StateKey, *routeCtx.actor)).value_or(false)
      ? HardRouteConstraint::Result{code_}
      : HardRouteConstraint::Result{};
  }

  /// Checks whether proposed activity insertion doesn't violate time windows.
  HardActivityConstraint::Result check(const InsertionRouteContext& routeCtx,
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

    //    |--- vehicle's operation time ---|  |--- prev or target or next ---|
    if (latestArrival < prev.detail.time.start || latestArrival < target.detail.time.start ||
        latestArrival < next.detail.time.start)
      return fail(code_);

    // |--- target ---| |--- prev ---|
    if (target.detail.time.end < prev.detail.time.start) return fail(code_);


    // |--- prev ---| |--- next ---| |- earliest arrival of vehicle
    auto arrTimeAtNext = actCtx.departure +
      transport_->duration(actor.vehicle->profile, prev.detail.location, nextActLocation, actCtx.departure);
    if (arrTimeAtNext > latestArrTimeAtNextAct) return fail(code_);


    //|--- next ---| |--- target ---|
    if (target.detail.time.start > next.detail.time.end) return stop(code_);


    auto arrTimeAtNewAct = actCtx.departure +
      transport_->duration(actor.vehicle->profile, prev.detail.location, target.detail.location, actCtx.departure);

    auto endTimeAtNewAct = std::max(arrTimeAtNewAct, target.detail.time.start)  //
      + activity_->duration(actor, target, arrTimeAtNewAct);

    std::int64_t time =
      transport_->duration(actor.vehicle->profile, target.detail.location, nextActLocation, latestArrTimeAtNextAct) -
      activity_->duration(actor, target, arrTimeAtNewAct);

    std::int64_t latestArrTimeAtNewAct = std::min<std::int64_t>(
      target.detail.time.end, static_cast<std::int64_t>(latestArrTimeAtNextAct) - static_cast<std::int64_t>(time));

    // |--- latest arrival of vehicle @target ---| |--- vehicle's arrival @target ---|
    if (static_cast<std::int64_t>(arrTimeAtNewAct) > latestArrTimeAtNewAct) return stop(code_);


    if (next.type == solution::Activity::Type::End && !actor.detail.end.has_value()) return success();


    auto arrTimeAtNextAct = endTimeAtNewAct +
      transport_->duration(actor.vehicle->profile, target.detail.location, nextActLocation, endTimeAtNewAct);

    //  |--- latest arrival of vehicle @next ---| |--- vehicle's arrival @next ---|
    return arrTimeAtNextAct > latestArrTimeAtNextAct ? stop(code_) : success();
  }

private:
  int code_;
  std::unordered_map<std::string, std::pair<models::common::Timestamp, models::common::Location>> keys_;
  std::shared_ptr<const models::costs::TransportCosts> transport_;
  std::shared_ptr<const models::costs::ActivityCosts> activity_;
};
}
