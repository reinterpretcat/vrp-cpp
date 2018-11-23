#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionProgress.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/problem/Service.hpp"

namespace vrp::algorithms::construction {

struct ShipmentInsertionEvaluator final {
  explicit ShipmentInsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                                      std::shared_ptr<const models::costs::ActivityCosts> activityCosts,
                                      std::shared_ptr<InsertionConstraint> constraint) :
    constraint_(std::move(constraint)) {}

  InsertionResult evaluate(const std::shared_ptr<const models::problem::Shipment>& shipment,
                           const InsertionRouteContext& ctx,
                           const InsertionProgress& progress) const {
    return InsertionResult{ranges::emplaced_index<1>, InsertionFailure{0}};
  }

private:
  std::shared_ptr<const InsertionConstraint> constraint_;
};

}  // namespace vrp::algorithms::construction