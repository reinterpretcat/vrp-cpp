#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "models/common/Location.hpp"
#include "models/common/Timestamp.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "utils/extensions/Hash.hpp"

#include <algorithm>
#include <optional>
#include <range/v3/all.hpp>
#include <string>
#include <utility>

namespace vrp::algorithms::construction {

/// Checks whether vehicle can serve activity taking into account their time windows.
struct VehicleActivityTiming final : public HardActivityConstraint {
  inline static const std::string StateKey = "vehicle_activity_timing";

  VehicleActivityTiming(std::shared_ptr<const models::problem::Fleet> fleet,
                        std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                        std::shared_ptr<const models::costs::ActivityCosts> activityCosts,
                        int code = 1) :
    code_(code),
    keyMapping_(), transportCosts_(std::move(transportCosts)), activityCosts_(std::move(activityCosts)) {
    // TODO consider driver as well
    ranges::for_each(fleet->vehicles(), [&](const auto& v) { keyMapping_[v->id] = key(*v); });
  }

  /// Accept route and updates its insertion state.
  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {
    using namespace ranges;
    const auto& stateKey = keyMapping_.find(route.actor.vehicle->id)->second;
    auto init = std::pair{route.end->time.end, route.end->location};
    ranges::accumulate(view::reverse(route.tour.activities()), init, [&](const auto& acc, const auto& act) {
      auto [endTime, location] = acc;
      auto potentialLatest = endTime - transportCosts_->duration(route.actor, act->location, location, endTime) -
        activityCosts_->duration(route.actor, *act, endTime);

      auto latestArrivalTime = std::min(act->time.end, potentialLatest);
      if (latestArrivalTime < act->time.start) state.put<bool>(stateKey, true);

      state.put<models::common::Timestamp>(stateKey, *act, latestArrivalTime);

      return std::pair{latestArrivalTime, act->location};
    });
  }

  /// Checks whether proposed insertion doesn't violate time windows.
  std::optional<std::tuple<bool, int>> check(const InsertionRouteContext& routeCtx,
                                             const InsertionActivityContext& actCtx) const override {
    using namespace vrp::models;
    using namespace vrp::models::common;

    // TODO check switch feasibility

    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = *actCtx.next;

    auto nextActLocation = next.location;
    auto latestArrival = routeCtx.actor->vehicle->time.end;
    auto latestArrTimeAtNextAct = next.type == solution::Activity::Type::End
      ? routeCtx.actor->vehicle->time.end
      : routeCtx.state->get<Timestamp>(StateKey, next).value_or(next.time.end);

    //    |--- vehicle's operation time ---|  |--- prev or target or next ---|
    if (latestArrival < prev.time.start || latestArrival < target.time.start || latestArrival < next.time.start)
      return fail();

    // |--- target ---| |--- prev ---|
    if (target.time.end < prev.time.start) return fail();


    // |--- prevAct ---| |--- nextAct ---| |- earliest arrival of vehicle
    auto arrTimeAtNext =
      actCtx.departure + transportCosts_->duration(*routeCtx.actor, prev.location, next.location, actCtx.departure);
    if (arrTimeAtNext > latestArrTimeAtNextAct) return fail();


    //|--- nextAct ---| |--- newAct ---|
    if (target.time.start > next.time.end) return fail();


    auto arrTimeAtNewAct = actCtx.departure  //
      + transportCosts_->duration(*routeCtx.actor, prev.location, target.location, actCtx.departure);

    auto endTimeAtNewAct = std::max(arrTimeAtNewAct, target.time.start)  //
      + activityCosts_->duration(*routeCtx.actor, target, arrTimeAtNewAct);

    auto time = transportCosts_->duration(*routeCtx.actor, target.location, nextActLocation, latestArrTimeAtNextAct)  //
      - activityCosts_->duration(*routeCtx.actor, target, arrTimeAtNewAct);

    auto latestArrTimeAtNewAct = std::min(target.time.end, latestArrTimeAtNextAct - time);

    // |--- latest arrival of vehicle @newAct ---| |--- vehicle's arrival @newAct ---|
    if (arrTimeAtNewAct > latestArrTimeAtNewAct) return stop();


    if (next.type == solution::Activity::Type::End && routeCtx.actor->vehicle->end.has_value()) return success();


    auto arrTimeAtNextAct =
      endTimeAtNewAct + transportCosts_->duration(*routeCtx.actor, target.location, nextActLocation, endTimeAtNewAct);

    //  |--- latest arrival of vehicle @nextAct ---| |--- vehicle's arrival @nextAct ---|
    return arrTimeAtNextAct > latestArrTimeAtNextAct ? stop() : success();
  }

private:
  std::optional<std::tuple<bool, int>> success() const { return {}; }
  std::optional<std::tuple<bool, int>> fail() const { return {{true, code_}}; }
  std::optional<std::tuple<bool, int>> stop() const { return {{true, -1}}; }

  std::string key(const models::problem::Vehicle& v) const {
    using namespace vrp::utils;
    using namespace vrp::models::common;
    auto hash = size_t{0} | hash_combine<Timestamp>{v.time.start} |  //
      hash_combine<Timestamp>{v.time.end} | hash_combine<Location>{v.start} |
      hash_combine<Location>{v.end.value_or(std::numeric_limits<std::uint64_t>::max())};
    return StateKey + std::to_string(hash);
  }

  int code_;
  std::unordered_map<std::string, std::string> keyMapping_;
  std::shared_ptr<const models::costs::TransportCosts> transportCosts_;
  std::shared_ptr<const models::costs::ActivityCosts> activityCosts_;
};
}
