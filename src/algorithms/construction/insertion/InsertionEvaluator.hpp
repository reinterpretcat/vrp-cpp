#pragma once

#include "algorithms/construction/insertion/InsertionContext.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"

#include <variant>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator {
  using Result = std::variant<InsertionResult::Success, InsertionResult::Failure>;

  /// Evaluates possibility to preform insertion from given insertion context.
  virtual Result evaluate(const InsertionContext& ctx, double bestKnownCost) = 0;

  virtual ~InsertionEvaluator() = default;
};

}  // namespace vrp::algorithms::construction