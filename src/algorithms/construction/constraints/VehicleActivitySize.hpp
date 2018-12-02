#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/States.hpp"

namespace vrp::algorithms::construction {

/// Checks whether vehicle can handle activity of given size.
struct VehicleActivitySize final
  : public HardRouteConstraint
  , public HardActivityConstraint {
  inline static const std::string StateKey = "size";

  /// Accept route and updates its insertion state.
  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {
    // TODO
  }

  /// Checks whether proposed vehicle can be used within route without violating size constraints.
  HardRouteConstraint::Result check(const InsertionRouteContext& routeCtx,
                                    const HardRouteConstraint::Job&) const override {
    // TODO
    return {};
  }

  /// Checks whether proposed activity insertion doesn't violate size constraints.
  HardActivityConstraint::Result check(const InsertionRouteContext& routeCtx,
                                       const InsertionActivityContext& actCtx) const override {
    // TODO
    return {};
  }
};
}
