#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "models/problem/Job.hpp"

#include <cmath>
#include <range/v3/all.hpp>

namespace vrp::algorithms::refinement {

/// "Adjusted string removal" strategy based on "Slack Induction by String Removals for
/// Vehicle Routing Problems" by Jan Christiaens, Greet Vanden Berghe.
/// Some definitions from paper:
///     String is a sequence of consecutive nodes in a tour.
///     Cardinality is the number of customers included in a string or tour.
struct RemoveAdjustedString {
  /// Specifies max removed string cardinality for specific tour.
  int lmax = 10;

  /// Specifies average number of removed customers.
  int cavg = 10;

  /// Identifies jobs to be ruined from given solution.
  ranges::any_view<models::problem::Job> operator()(const RefinementContext& ctx,
                                                    const models::Solution& solution) const {
    /// Equation 5: max removed string cardinality for each tour
    auto lsmax = std::min(lmax, tourCardinality(solution));

    /// Equation 6: max number of strings
    auto ksmax = 4 * cavg / (1 + lsmax) - 1;

    /// Equation 7:
    auto ks = 0;
  }

private:
  /// Calculates average tour cardinality.
  int tourCardinality(const models::Solution& solution) const {
    return static_cast<int>(
      std::round(ranges::accumulate(
                   solution.routes, 0.0, [](const double acc, const auto& r) { return acc + r->tour.sizes().second; }) /
                 solution.routes.size()));
  }
};
}
