#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"
#include "algorithms/construction/insertion/evaluators/ShipmentInsertionEvaluator.hpp"

#include <variant>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {
  explicit InsertionEvaluator(std::shared_ptr<const InsertionConstraint> constraint) :
    serviceInsertionEvaluator(constraint), shipmentInsertionEvaluator(constraint) {}

  /// Evaluates possibility to preform insertion from given insertion context.
  InsertionResult evaluate(const models::problem::Job& job, const InsertionRouteContext& ctx, double bestKnownCost) {
    // TODO insert start/end?
    return job.visit(ranges::overload(
      [&](const auto& service) { return serviceInsertionEvaluator.evaluate(service, ctx, bestKnownCost); },
      [&](const auto& shipment) { return shipmentInsertionEvaluator.evaluate(shipment, ctx, bestKnownCost); }));
  }

private:
  const ServiceInsertionEvaluator serviceInsertionEvaluator;
  const ShipmentInsertionEvaluator shipmentInsertionEvaluator;
};

}  // namespace vrp::algorithms::construction