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

  /// Specifies actor which should be used.
  models::solution::Route::Actor actor;

  /// Specifies route where insertion happens.
  InsertionRouteContext::RouteState route;

  /// Specifies new vehicle departure time.
  models::common::Timestamp departure;
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
get_cheapest(const InsertionResult& left, const InsertionResult& right) {
  return utils::mono_result<InsertionResult>(right.visit(ranges::overload(
    [&](const InsertionSuccess& success) {
      if (left.index() == 1) return right;
      return ranges::get<0>(left).cost > success.cost ? make_result_success(success) : left;
    },
    [&](const InsertionFailure& failure) {
      return left.index() == 1 ? make_result_failure(failure.constraint) : left;
    })));
}

}  // namespace vrp::algorithms::construction
