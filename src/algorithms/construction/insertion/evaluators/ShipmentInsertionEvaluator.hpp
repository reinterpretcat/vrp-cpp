#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "models/problem/Service.hpp"

namespace vrp::algorithms::construction {

struct ShipmentInsertionEvaluator final {
  explicit ShipmentInsertionEvaluator(std::shared_ptr<const InsertionConstraint> constraint) :
    constraint_(std::move(constraint)) {}

  InsertionResult::Variant evaluate(const models::problem::Shipment& shipment,
                                    const InsertionContext& ctx,
                                    double bestKnownCost) const {
    // TODO
  }

private:
  std::shared_ptr<const InsertionConstraint> constraint_;
};

}  // namespace vrp::algorithms::construction