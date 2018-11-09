#pragma once

#include "algorithms/construction/insertion/InsertionEvaluator.hpp"

namespace vrp::algorithms::construction {

struct ShipmentInsertionEvaluator final : public InsertionEvaluator {
  InsertionEvaluator::Result evaluate(const InsertionContext& ctx, double bestKnownCost) override {}
};

}  // namespace vrp::algorithms::construction