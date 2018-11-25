#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "models/common/Cost.hpp"

#include <algorithm>
#include <memory>

namespace vrp::algorithms::construction {

/// Implements default scoring function for regret insertion.
struct ScoreFunction final {
  models::common::Cost score(const InsertionSuccess& first, const InsertionSuccess& second) const {
    return second.cost - first.cost;
  }
};

/// Insertion based on regret approach.
/// Basically calculates the insertion cost of the first best and the second best alternative. The score is then
/// calculated as difference between second and first best, plus additional scoring variables that can defined
/// in score function.
/// The idea is that if the cost of the second best alternative is way higher than the first best, it seems to be
/// important to insert this customer immediately. If difference is not that high, it might not impact solution
/// if this customer is inserted later.
template <typename ScoreFunc>
struct RegretInsertion final : InsertionHeuristic<RegretInsertion<ScoreFunc>> {
  struct Score final {
    /// Insertion score value.
    models::common::Cost value;
    /// Insertion result.
    InsertionSuccess result;
  };

  explicit RegretInsertion(const std::shared_ptr<const InsertionEvaluator>& evaluator) : evaluator_(evaluator) {}

  InsertionContext analyze(const InsertionContext& ctx) const {
    // TODO

    return {};
  }

 private:
  std::shared_ptr<const InsertionEvaluator> evaluator_;
};
}