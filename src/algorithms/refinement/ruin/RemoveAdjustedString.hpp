#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "models/extensions/solution/Selectors.hpp"
#include "models/problem/Job.hpp"

#include <cmath>
#include <optional>
#include <range/v3/all.hpp>
#include <tuple>

namespace vrp::algorithms::refinement {

/// "Adjusted string removal" strategy based on "Slack Induction by String Removals for
/// Vehicle Routing Problems" (aka SISR) by Jan Christiaens, Greet Vanden Berghe.
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
    auto job = models::solution::select_job{}(solution.routes, *ctx.random);
    if (!job) return ranges::view::empty<models::problem::Job>();

    auto [lsmax, ksmax, ks] = initialParams(ctx, solution);
  }

private:
  /// Calculates initial parameters from paper using 5,6,7 equations.
  std::tuple<double, double, int> initialParams(const RefinementContext& ctx, const models::Solution& solution) const {
    /// Equation 5: max removed string cardinality for each tour
    double lsmax = std::min(static_cast<double>(lmax), avgTourCardinality(solution));

    /// Equation 6: max number of strings
    double ksmax = 4 * cavg / (1 + lsmax) - 1;

    /// Equation 7: number of string to be removed
    int ks = static_cast<int>(std::floor(ctx.random->uniform<double>(1, ksmax + 1)));

    return {lsmax, ksmax, ks};
  }

  /// Calculates average tour cardinality.
  double avgTourCardinality(const models::Solution& solution) const {
    return std::round(ranges::accumulate(solution.routes,
                                         0.0,
                                         [](const double acc, const auto& r) { return acc + r->tour.sizes().second; }) /
                      solution.routes.size());
  }
};
}
