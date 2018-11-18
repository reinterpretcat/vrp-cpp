#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "models/common/Location.hpp"
#include "models/common/Timestamp.hpp"

#include <optional>
#include <utility>

namespace vrp::algorithms::construction {

/// Checks whether vehicle can serve activity taking into account their time windows.
struct VehicleActivityTiming final : public HardActivityConstraint {
  constexpr static int code = 1;

  /// Accept route and updates its state.
  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {}

  /// Checks whether insertion contexts don't violate specific constraints.
  std::optional<std::tuple<bool, int>> check(const InsertionRouteContext& routeCtx,
                                             const InsertionActivityContext& actCtx) const override {
    auto latestVehicleArrival = routeCtx.actor->vehicle->time.end;

    return {};
  }

private:
  auto endInfo(const InsertionRouteContext& routeCtx, const InsertionActivityContext& actCtx) const {
    using namespace vrp::models;
    using namespace vrp::models::common;

    Timestamp latestArrTimeAtNextAct;
    Location nextActLocation;
    if (actCtx.next->type == solution::Activity::Type::End) {
      latestArrTimeAtNextAct = routeCtx.actor->vehicle->time.end;
      nextActLocation = actCtx.next->location;
    } else {
    }

    return std::pair<Timestamp, Location>{};
  };
};
}
