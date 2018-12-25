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
/// Some definitions from the paper:
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
    auto jobs = std::make_shared<std::set<models::problem::Job, models::problem::compare_jobs>>();
    auto routes = std::make_shared<std::set<std::shared_ptr<models::solution::Route>>>();
    auto [lsmax, ks] = initialParams(ctx, solution);

    while (routes->size() != ks) {
      ranges::for_each(neighbors(ctx, solution), [=, &solution, lsmax = lsmax](const auto& j) {
        if (jobs->find(j) == jobs->end() && solution.unassignedJobs.find(j) == solution.unassignedJobs.end()) {
          ranges::for_each(solution.routes, [=, &ctx](const auto& r) {
            if (routes->find(r) == routes->end() && r->tour.has(j)) {
              /// Equations 8, 9: calculate cardinality of the string removed from the tour
              auto ltmax = std::min(static_cast<double>(r->tour.sizes().second), lsmax);
              auto lt = static_cast<int>(std::floor(ctx.random->uniform<double>(1, ltmax + 1)));

              removeSelected(ctx, *r, j, lt, *jobs);
              routes->insert(r);
            }
          });
        }
      });
    }

    return ranges::view::all(*jobs);
  }

private:
  /// Calculates initial parameters from paper using 5,6,7 equations.
  std::tuple<double, int> initialParams(const RefinementContext& ctx, const models::Solution& solution) const {
    /// Equation 5: max removed string cardinality for each tour
    double lsmax = std::min(static_cast<double>(lmax), avgTourCardinality(solution));

    /// Equation 6: max number of strings
    double ksmax = 4 * cavg / (1 + lsmax) - 1;

    /// Equation 7: number of string to be removed
    int ks = static_cast<int>(std::floor(ctx.random->uniform<double>(1, ksmax + 1)));

    return {lsmax, ks};
  }

  /// Returns all neighbours of seed job.
  ranges::any_view<models::problem::Job> neighbors(const RefinementContext& ctx,
                                                   const models::Solution& solution) const {
    auto seed = models::solution::select_job{}(solution.routes, *ctx.random);
    if (!seed) return ranges::view::empty<models::problem::Job>();

    auto [route, job] = seed.value();

    return ctx.problem->jobs->neighbors(route->actor->vehicle->profile, job, models::common::Timestamp{0});
  }

  /// Calculates average tour cardinality.
  double avgTourCardinality(const models::Solution& solution) const {
    return std::round(ranges::accumulate(solution.routes,
                                         0.0,
                                         [](const double acc, const auto& r) { return acc + r->tour.sizes().second; }) /
                      solution.routes.size());
  }

  /// Removes string for selected customer.
  void removeSelected(const RefinementContext& ctx,
                      const models::solution::Route& route,
                      const models::problem::Job& job,
                      int cardinality,
                      std::set<models::problem::Job, models::problem::compare_jobs>& jobs) const {
    if (ctx.random->uniform<int>(1, 2) == 1)
      removeSequentialString(ctx, route, job, cardinality, jobs);
    else
      removeSplitString(ctx, route, job, cardinality, jobs);
  }

  /// Remove sequential string.
  void removeSequentialString(const RefinementContext& ctx,
                    const models::solution::Route& route,
                    const models::problem::Job& job,
                    int cardinality,
                    std::set<models::problem::Job, models::problem::compare_jobs>& jobs) const {
    // TODO
  }

  /// Remove string with gap.
  void removeSplitString(const RefinementContext& ctx,
                         const models::solution::Route& route,
                         const models::problem::Job& job,
                         int cardinality,
                         std::set<models::problem::Job, models::problem::compare_jobs>& jobs) const {
    // TODO
  }
};
}
