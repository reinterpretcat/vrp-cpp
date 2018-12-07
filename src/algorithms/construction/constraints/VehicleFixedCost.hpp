#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"

namespace vrp::algorithms::construction {

/// Provides the way to apply extra cost on vehicle usage.
struct VehicleFixedCost final : public SoftRouteConstraint {
  double weight = 1;

  models::common::Cost check(const InsertionRouteContext& ctx, const Job& job) const override {
    return ctx.actor->vehicle != ctx.route.first->actor->vehicle ? ctx.actor->vehicle->costs.fixed * weight : 0;
  }
};
}