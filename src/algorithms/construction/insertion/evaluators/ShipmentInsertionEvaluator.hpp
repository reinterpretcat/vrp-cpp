#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "models/common/Cost.hpp"
#include "models/problem/Service.hpp"

namespace vrp::algorithms::construction {

struct ShipmentInsertionEvaluator final {
  explicit ShipmentInsertionEvaluator(std::shared_ptr<const InsertionConstraint> constraint) :
    constraint_(std::move(constraint)) {}

  InsertionResult evaluate(const std::shared_ptr<const models::problem::Shipment>& shipment,
                           const InsertionRouteContext& ctx,
                           const InsertionProgress& progress) const {
    //    return InsertionResult::Failure{};
  }

private:
  std::shared_ptr<const InsertionConstraint> constraint_;
};

}  // namespace vrp::algorithms::construction