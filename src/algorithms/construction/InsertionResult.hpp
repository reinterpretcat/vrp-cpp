#pragma once

#include "algorithms/construction/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/common/Timestamp.hpp"
#include "models/solution/Actor.hpp"
#include "models/solution/Route.hpp"
#include "models/solution/Tour.hpp"

#include <memory>
#include <range/v3/utility/variant.hpp>

namespace vrp::algorithms::construction {

/// Specifies insertion result needed to insert job into tour.
struct InsertionSuccess final {
  /// Specifies delta cost change for the insertion.
  models::common::Cost cost;

  /// Original job to be inserted.
  models::problem::Job job;

  /// Specifies activities within index where they have to be inserted.
  std::vector<std::pair<models::solution::Tour::Activity, size_t>> activities;

  /// Specifies route context where insertion happens.
  InsertionRouteContext context;
};

/// Specifies insertion failure.
struct InsertionFailure final {
  /// Failed constraint code.
  int constraint;
};

/// Specifies all possible insertion results.
using InsertionResult = ranges::variant<InsertionSuccess, InsertionFailure>;

/// Creates result which represents insertion failure.
inline InsertionResult
make_result_failure(int code = 0) {
  return InsertionResult{ranges::emplaced_index<1>, InsertionFailure{code}};
}

/// Creates result which represents insertion success.
inline InsertionResult
make_result_success(const InsertionSuccess& success) {
  return InsertionResult{ranges::emplaced_index<0>, success};
}

/// Compares two insertion results and returns the cheapest by cost.
inline InsertionResult
get_best_result(const InsertionResult& left, const InsertionResult& right) {
  // NOTE seems this approach is much faster than visit&overload
  if (right.index() == 0) {
    if (left.index() == 1) return right;
    return ranges::get<0>(left).cost > ranges::get<0>(right).cost ? make_result_success(ranges::get<0>(right)) : left;
  }
  return left.index() == 1 ? make_result_failure(ranges::get<1>(right).constraint) : left;
}

}  // namespace vrp::algorithms::construction
