#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "models/common/Cost.hpp"

#include <rxcpp/rx.hpp>

namespace vrp::algorithms::construction {

/// Cheapest insertion heuristic.
struct CheapestInsertion final : InsertionHeuristic<CheapestInsertion> {
  explicit CheapestInsertion(const InsertionEvaluator& evaluator) : evaluator_(evaluator) {}

  InsertionContext analyze(const InsertionContext& ctx) const {
    auto newCtx = InsertionContext(ctx);
    while (!newCtx.jobs.empty()) {
      // TODO use C++17 parallel algorithms instead of rxcpp once it has better runtime support
      insert(rxcpp::observable<>::iterate(newCtx.jobs)
               .map([&](const auto& job) { return evaluator_.evaluate(job, newCtx); })
               .reduce(make_result_failure(),
                       [](const auto& acc, const auto& result) { return get_cheapest(acc, result); },
                       [](const auto& res) { return res; })
               .as_blocking()
               .last(),
             newCtx);
    }
    return newCtx;
  }

private:
  const InsertionEvaluator evaluator_;
};
}