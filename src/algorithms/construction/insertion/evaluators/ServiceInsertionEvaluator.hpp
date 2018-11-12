#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "algorithms/construction/insertion/evaluators/JobInsertionEvaluator.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Service.hpp"

namespace vrp::algorithms::construction {

struct ServiceInsertionEvaluator final : private JobInsertionEvaluator {
  ServiceInsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                            std::shared_ptr<const InsertionConstraint> constraint) :
    JobInsertionEvaluator(transportCosts),
    constraint_(std::move(constraint)) {}

  InsertionResult evaluate(const std::shared_ptr<const models::problem::Service>& service,
                           const InsertionRouteContext& ctx,
                           models::common::Cost bestKnown) const {
    auto activity = models::solution::build_activity{}            //
                      .withJob(models::problem::as_job(service))  //
                      .owned();

    auto error = constraint_->hard(ctx, ranges::view::single(activity));
    if (error.has_value()) { return {ranges::emplaced_index<1>, InsertionFailure{error.value()}}; }

    auto additionalCosts = vehicleSwitchCost(ctx);

    return {ranges::emplaced_index<1>, InsertionFailure{}};
  }

private:
  std::shared_ptr<const InsertionConstraint> constraint_;
};

}  // namespace vrp::algorithms::construction
