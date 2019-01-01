#pragma once

#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"

namespace vrp::algorithms::construction {

/// Selects best result with blinks where ratio is defined by nominator / denominator.
template<int Nominator, int Denominator>
struct select_insertion_with_blinks final {
  constexpr static double ratio = double(Nominator) / Denominator;

  const InsertionContext& ctx;

  InsertionResult operator()(const InsertionResult& left, const InsertionResult& right) const {
    return get_best_result(left, right);
  }
};

/// Specifies insertion with blinks heuristic.
template<int Nominator = 1, int Denominator = 100>
using BlinkInsertion = InsertionHeuristic<InsertionEvaluator, select_insertion_with_blinks<Nominator, Denominator>>;
}