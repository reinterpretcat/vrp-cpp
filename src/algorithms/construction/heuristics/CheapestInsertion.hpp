#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "models/common/Cost.hpp"

#include <pstl/execution>
#include <pstl/numeric>

namespace vrp::algorithms::construction {

/// Selects always best insertion result.
struct select_insertion_greedy final {
  InsertionResult operator()(const InsertionResult& left, const InsertionResult& right) const {
    return get_best_result(left, right);
  }
};

/// Cheapest insertion heuristic.
/// Evaluator template param specifies the logic responsible for job insertion
/// evaluation: where is job's insertion point.
/// Selector template param specifies the logic which selects job to be inserted.
template<typename Evaluator, typename Selector = select_insertion_greedy>
struct CheapestInsertion final : InsertionHeuristic<CheapestInsertion<Evaluator, Selector>> {
  explicit CheapestInsertion(const Evaluator& evaluator) : evaluator_(evaluator) {}

  void accept(const InsertionSuccess& success) { evaluator_.accept(success.route); }

  InsertionContext insert(const InsertionContext& ctx) const {
    auto newCtx = InsertionContext(ctx);
    auto selector = Selector{};
    while (!newCtx.jobs.empty()) {
      InsertionHeuristic<CheapestInsertion<Evaluator, Selector>>::insert(
        std::transform_reduce(pstl::execution::par,
                              newCtx.jobs.begin(),
                              newCtx.jobs.end(),
                              make_result_failure(),
                              [&](const auto& acc, const auto& result) { return selector(acc, result); },
                              [&](const auto& job) { return evaluator_.evaluate(job, newCtx); }),
        newCtx);
    }
    return std::move(newCtx);
  }

private:
  const Evaluator evaluator_;
};
}