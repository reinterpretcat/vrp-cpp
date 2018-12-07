#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "models/common/Cost.hpp"

#include <pstl/execution>
#include <pstl/numeric>

namespace vrp::algorithms::construction {

/// Cheapest insertion heuristic.
template<typename Evaluator>
struct CheapestInsertion final : InsertionHeuristic<CheapestInsertion<Evaluator>> {
  explicit CheapestInsertion(const Evaluator& evaluator) : evaluator_(evaluator) {}

  void accept(const InsertionSuccess& success) { evaluator_.accept(success.route); }

  InsertionContext insert(const InsertionContext& ctx) const {
    auto newCtx = InsertionContext(ctx);
    while (!newCtx.jobs.empty()) {
      InsertionHeuristic<CheapestInsertion<Evaluator>>::insert(
        std::transform_reduce(pstl::execution::par,
                              newCtx.jobs.begin(),
                              newCtx.jobs.end(),
                              make_result_failure(),
                              [](const auto& acc, const auto& result) { return get_cheapest(acc, result); },
                              [&](const auto& job) { return evaluator_.evaluate(job, newCtx); }),
        newCtx);
    }
    return std::move(newCtx);
  }

private:
  const Evaluator evaluator_;
};
}