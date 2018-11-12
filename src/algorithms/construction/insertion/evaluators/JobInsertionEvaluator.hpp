#pragma once

#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/TransportCosts.hpp"

namespace vrp::algorithms::construction {

struct JobInsertionEvaluator {
  explicit JobInsertionEvaluator(const std::shared_ptr<models::costs::TransportCosts> transportCosts) :
    transportCosts_(transportCosts) {}

  virtual ~JobInsertionEvaluator() = default;

protected:
  /// Estimates vehicle switch cost if when new vehicle is used.
  models::common::Cost vehicleSwitchCost(const InsertionRouteContext& ctx) const {
    double delta_access = 0.0;
    double delta_egress = 0.0;
    auto currentRoute = ctx.route;
    auto newVehicleDepartureTime = ctx.time;

    
    return 0;
  }

private:
  std::shared_ptr<models::costs::TransportCosts> transportCosts_;
};
}
