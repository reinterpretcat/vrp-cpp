#pragma once

#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"

namespace vrp::algorithms::construction {

/// Selects always best insertion result.
struct select_insertion_greedy final {
  InsertionResult operator()(const InsertionResult& left, const InsertionResult& right) const {
    return get_best_result(left, right);
  }
};

/// Specifies cheapest insertion heuristic.
using CheapestInsertion = InsertionHeuristic<InsertionEvaluator, select_insertion_greedy>;
}