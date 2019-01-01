#pragma once

#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"

#include <mutex>

namespace vrp::algorithms::construction {

/// Selects best result with blinks where ratio is defined by nominator / denominator.
template<int Nominator, int Denominator>
struct select_insertion_with_blinks final {
  constexpr static double ratio = double(Nominator) / Denominator;

  explicit select_insertion_with_blinks(const InsertionContext& ctx) : ctx_(ctx), lock_() {}

  InsertionResult operator()(const InsertionResult& left, const InsertionResult& right) const {
    return isBlink() ? left : get_best_result(left, right);
  }

private:
  /// Checks whether new job insertion result should be ignored.
  bool isBlink() const {
    std::lock_guard<std::mutex> lock(lock_);
    return ctx_.random->uniform<double>(0, 1) < ratio;
  }
  const InsertionContext& ctx_;
  mutable std::mutex lock_;
};

/// Specifies insertion with blinks heuristic.
template<int Nominator = 1, int Denominator = 100>
using BlinkInsertion = InsertionHeuristic<InsertionEvaluator, select_insertion_with_blinks<Nominator, Denominator>>;
}