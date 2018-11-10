#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "models/problem/Service.hpp"

#include <models/extensions/solution/Factories.hpp>

namespace vrp::algorithms::construction {

struct ServiceInsertionEvaluator final {
  explicit ServiceInsertionEvaluator(std::shared_ptr<const InsertionConstraint> constraint) :
    constraint_(std::move(constraint)) {}

  InsertionResult::Variant evaluate(const std::shared_ptr<const models::problem::Service>& service,
                                    const InsertionContext& ctx,
                                    double bestKnownCost) const {
    auto activity = models::solution::build_activity{}.withJob(service).withSchedule({0, 0}).owned();

    auto fulfilled = constraint_->hard(ctx, ranges::view::single(activity));

    return InsertionResult::Failure{};
  }

private:
  std::shared_ptr<const InsertionConstraint> constraint_;
};

}  // namespace vrp::algorithms::construction