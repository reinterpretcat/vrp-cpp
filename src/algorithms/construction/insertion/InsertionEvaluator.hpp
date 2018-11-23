#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionProgress.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"
#include "algorithms/construction/insertion/evaluators/ShipmentInsertionEvaluator.hpp"
#include "utils/extensions/Variant.hpp"

#include <variant>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {
  explicit InsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                              std::shared_ptr<const models::costs::ActivityCosts> activityCosts,
                              std::shared_ptr<InsertionConstraint> constraint) :
    serviceInsertionEvaluator(transportCosts, activityCosts, constraint),
    shipmentInsertionEvaluator(transportCosts, activityCosts, constraint) {}

  /// Evaluates possibility to preform insertion from given insertion context.
  InsertionResult evaluate(const models::problem::Job& job,
                           const InsertionRouteContext& ctx,
                           const InsertionProgress& progress) {
    // TODO insert start/end?

    return utils::mono_result<InsertionResult>(job.visit(ranges::overload(
      [&](const std::shared_ptr<const models::problem::Service>& service) {
        return serviceInsertionEvaluator.evaluate(service, ctx, progress);
      },
      [&](const std::shared_ptr<const models::problem::Shipment>& shipment) {
        return shipmentInsertionEvaluator.evaluate(shipment, ctx, progress);
      })));
  }

private:
  const ServiceInsertionEvaluator serviceInsertionEvaluator;
  const ShipmentInsertionEvaluator shipmentInsertionEvaluator;
};

}  // namespace vrp::algorithms::construction