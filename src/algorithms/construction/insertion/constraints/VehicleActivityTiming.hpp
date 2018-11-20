#pragma once

#include "algorithms/construction/extensions/States.hpp"
#include "algorithms/construction/insertion/InsertionConstraint.hpp"
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
  inline static const std::string StateKey = "operation_time";

  VehicleActivityTiming(std::shared_ptr<const models::problem::Fleet> fleet,
                        std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                        std::shared_ptr<const models::costs::ActivityCosts> activityCosts,
                        int code = 1) :
    code_(code),
    mapping_(), keys_(), transportCosts_(std::move(transportCosts)), activityCosts_(std::move(activityCosts)) {
    // TODO consider driver as well
    ranges::for_each(fleet->vehicles(), [&](const auto& v) {
      auto key = vehicleKey(StateKey, *v);
      mapping_[v->id] = key;
      if (keys_.find(key) == keys_.end()) { keys_[key] = std::make_pair(v->time.end, v->end.value_or(v->start)); }
    });
  }

  /// Accept route and updates its insertion state.
  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {
    using namespace ranges;

    ranges::for_each(ranges::view::all(keys_), [&](const auto& pair) {
      const auto& stateKey = pair.first;
      auto init = std::pair{pair.second.first, pair.second.second};

      ranges::accumulate(view::reverse(route.tour.activities()), init, [&](const auto& acc, const auto& act) {
        const auto& [endTime, prevLoc] = acc;

        auto potentialLatest = endTime - transportCosts_->duration(route.actor, act->location, prevLoc, endTime) -
          activityCosts_->duration(route.actor, *act, endTime);

        auto latestArrivalTime = std::min(act->time.end, potentialLatest);
        if (latestArrivalTime < act->time.start) state.put<bool>(stateKey, true);

        state.put<models::common::Timestamp>(stateKey, *act, latestArrivalTime);

        return std::make_pair(latestArrivalTime, act->location);
      });
    });
  }

  /// Checks whether proposed vehicle can be used within route without violating time windows.
  HardRouteConstraint::Result check(const InsertionRouteContext& routeCtx,
                                    const HardRouteConstraint::Activities&) const override {
    return routeCtx.state->get<bool>(mapping_.find(routeCtx.actor->vehicle->id)->second).value_or(false)
      ? HardRouteConstraint::Result{code_}
      : HardRouteConstraint::Result{};
  }

  /// Checks whether proposed activity insertion doesn't violate time windows.
  HardActivityConstraint::Result check(const InsertionRouteContext& routeCtx,
                                       const InsertionActivityContext& actCtx) const override {
    using namespace vrp::models;
    using namespace vrp::models::common;

    const auto& actor = routeCtx.actor;
    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = *actCtx.next;

    auto latestArrival = routeCtx.actor->vehicle->time.end;
    auto nextActLocation =
      next.type == solution::Activity::Type::End ? actor->vehicle->end.value_or(next.location) : next.location;
    auto latestArrTimeAtNextAct = next.type == solution::Activity::Type::End
      ? actor->vehicle->time.end
      : routeCtx.state->get<Timestamp>(mapping_.find(actor->vehicle->id)->second, next).value_or(next.time.end);

    //    |--- vehicle's operation time ---|  |--- prev or target or next ---|
    if (latestArrival < prev.time.start || latestArrival < target.time.start || latestArrival < next.time.start)
      return fail();

    // |--- target ---| |--- prev ---|
    if (target.time.end < prev.time.start) return fail();


    // |--- prev ---| |--- next ---| |- earliest arrival of vehicle
    auto arrTimeAtNext =
      actCtx.departure + transportCosts_->duration(*actor, prev.location, nextActLocation, actCtx.departure);
    if (arrTimeAtNext > latestArrTimeAtNextAct) return fail();


    //|--- next ---| |--- target ---|
    if (target.time.start > next.time.end) return fail();


    auto arrTimeAtNewAct = actCtx.departure  //
      + transportCosts_->duration(*actor, prev.location, target.location, actCtx.departure);

    auto endTimeAtNewAct = std::max(arrTimeAtNewAct, target.time.start)  //
      + activityCosts_->duration(*actor, target, arrTimeAtNewAct);

    std::int64_t time = transportCosts_->duration(*actor, target.location, nextActLocation, latestArrTimeAtNextAct)  //
      - activityCosts_->duration(*actor, target, arrTimeAtNewAct);

    std::int64_t latestArrTimeAtNewAct = std::min<std::int64_t>(
      target.time.end, static_cast<std::int64_t>(latestArrTimeAtNextAct) - static_cast<std::int64_t>(time));

    // |--- latest arrival of vehicle @target ---| |--- vehicle's arrival @target ---|
    if (static_cast<std::int64_t>(arrTimeAtNewAct) > latestArrTimeAtNewAct) return stop();


    if (next.type == solution::Activity::Type::End && !actor->vehicle->end.has_value()) return success();


    auto arrTimeAtNextAct =
      endTimeAtNewAct + transportCosts_->duration(*actor, target.location, nextActLocation, endTimeAtNewAct);

    //  |--- latest arrival of vehicle @next ---| |--- vehicle's arrival @next ---|
    return arrTimeAtNextAct > latestArrTimeAtNextAct ? stop() : success();
  }

private:
  HardActivityConstraint::Result success() const { return {}; }
  HardActivityConstraint::Result fail() const { return {{true, code_}}; }
  HardActivityConstraint::Result stop() const { return {{false, code_}}; }

  int code_;
  std::unordered_map<std::string, std::string> mapping_;
  std::unordered_map<std::string, std::pair<models::common::Timestamp, models::common::Location>> keys_;
  std::shared_ptr<const models::costs::TransportCosts> transportCosts_;
  std::shared_ptr<const models::costs::ActivityCosts> activityCosts_;
};
}
