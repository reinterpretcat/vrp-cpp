#pragma once

#include "algorithms/construction/insertion/InsertionEvaluator.hpp"
#include "models/problem/Service.hpp"

namespace vrp::algorithms::construction {

struct ShipmentInsertionEvaluator final {
  InsertionEvaluator::Result evaluate(const models::problem::Shipment& shipment,
                                      const InsertionContext& ctx,
                                      double bestKnownCost) {
    // TODO
  }
};

}  // namespace vrp::algorithms::construction