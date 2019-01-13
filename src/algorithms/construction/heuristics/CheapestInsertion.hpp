#pragma once

#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"

namespace vrp::algorithms::construction {

/// Selects always best insertion result.
struct select_insertion_result_greedy final {
  explicit select_insertion_result_greedy(const InsertionContext& ctx) {}

  InsertionResult operator()(const InsertionResult& left, const InsertionResult& right) const {
    return get_best_result(left, right);
  }
};

/// Selects jobs range greedy.
struct select_insertion_range_greedy final {
  auto operator()(const InsertionContext& ctx) const { return std::pair(ctx.jobs.begin(), ctx.jobs.end()); }
};

/// Specifies cheapest insertion heuristic.
using CheapestInsertion =
  InsertionHeuristic<InsertionEvaluator, select_insertion_range_greedy, select_insertion_result_greedy>;
}