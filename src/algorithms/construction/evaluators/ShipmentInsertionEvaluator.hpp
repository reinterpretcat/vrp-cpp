#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionResult.hpp"

namespace vrp::algorithms::construction {

struct ShipmentInsertionEvaluator final {
  InsertionResult evaluate(const std::shared_ptr<const models::problem::Shipment>& shipment,
                           const InsertionRouteContext& ctx,
                           const InsertionConstraint& constraint,
                           const InsertionProgress& progress) const {
    return InsertionResult{ranges::emplaced_index<1>, InsertionFailure{0}};
  }
};

}  // namespace vrp::algorithms::construction