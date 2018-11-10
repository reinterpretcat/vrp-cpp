#pragma once

#include "algorithms/construction/insertion/InsertionContext.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"
#include "algorithms/construction/insertion/evaluators/ShipmentInsertionEvaluator.hpp"
#include "models/extensions/problem/Adaptors.hpp"

#include <variant>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {
  using Result = std::variant<InsertionResult::Success, InsertionResult::Failure>;

  /// Evaluates possibility to preform insertion from given insertion context.
  Result evaluate(const InsertionContext& ctx, double bestKnownCost) {
    return models::problem::visit_job<Result>(
      [&](const auto& service) { return serviceInsertionEvaluator.evaluate(service, ctx, bestKnownCost); },
      [&](const auto& shipment) { return shipmentInsertionEvaluator.evaluate(shipment, ctx, bestKnownCost); },
      *ctx.job);
  }

private:
  ServiceInsertionEvaluator serviceInsertionEvaluator;
  ShipmentInsertionEvaluator shipmentInsertionEvaluator;
};

}  // namespace vrp::algorithms::construction