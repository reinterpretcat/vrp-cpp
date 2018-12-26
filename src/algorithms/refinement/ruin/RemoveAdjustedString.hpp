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

  /// Preserved customers ratio.
  double alpha = 0.01;

  /// Ruins jobs from given solution.
  void operator()(const RefinementContext& ctx, models::Solution& solution) const {
    using namespace ranges;
    auto jobs = std::make_shared<std::set<models::problem::Job, models::problem::compare_jobs>>();
    auto routes = std::make_shared<std::set<std::shared_ptr<models::solution::Route>>>();
    auto [lsmax, ks] = limits(ctx, solution);

    while (routes->size() != ks) {
      ranges::for_each(neighbors(ctx, solution), [=, &solution, lsmax = lsmax](const auto& j) {
        if (jobs->find(j) == jobs->end() && solution.unassignedJobs.find(j) == solution.unassignedJobs.end()) {
          ranges::for_each(solution.routes, [=, &ctx](const auto& r) {
            if (routes->find(r) == routes->end() && r->tour.has(j)) {
              /// Equations 8, 9: calculate cardinality of the string removed from the tour
              auto ltmax = std::min(static_cast<double>(r->tour.sizes().second), lsmax);
              auto lt = static_cast<int>(std::floor(ctx.random->uniform<double>(1, ltmax + 1)));

              routes->insert(r);
              action::insert(*jobs, removeSelected(ctx, r->tour, j, lt) | view::for_each([&](const auto& j) {
                                      auto toBeRemoved = ctx.locked.find(j) == ctx.locked.end();
                                      if (toBeRemoved) r->tour.remove(j);
                                      return ranges::yield_if(toBeRemoved, j);
                                    }));
            }
          });
        }
      });
    }
  }

private:
  /// Calculates initial parameters from paper using 5,6,7 equations.
  std::tuple<double, int> limits(const RefinementContext& ctx, const models::Solution& solution) const {
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

  // region String removal

  /// Removes string for selected customer.
  ranges::any_view<models::problem::Job> removeSelected(const RefinementContext& ctx,
                                                        const models::solution::Tour& tour,
                                                        const models::problem::Job& job,
                                                        int cardinality) const {
    auto index = static_cast<int>(tour.index(job));
    return ctx.random->isHeadsNotTails() ? removeSequentialString(ctx, tour, index, cardinality)
                                         : removePreservedString(ctx, tour, index, cardinality);
  }

  /// Remove sequential string.
  ranges::any_view<models::problem::Job> removeSequentialString(const RefinementContext& ctx,
                                                                const models::solution::Tour& tour,
                                                                int index,
                                                                int cardinality) const {
    auto bounds = lowerBounds(cardinality, static_cast<int>(tour.sizes().second), index) | ranges::to_vector;
    auto start = bounds.at(ctx.random->uniform<int>(0, bounds.size() - 1));

    return ranges::view::for_each(ranges::view::ints(start, start + cardinality), [&tour](int i) {
      auto j = tour.get(static_cast<size_t>(i))->job;
      return ranges::yield_if(j.has_value(), j.value());
    });
  }

  /// Remove string with preserved customers.
  ranges::any_view<models::problem::Job> removePreservedString(const RefinementContext& ctx,
                                                               const models::solution::Tour& tour,
                                                               int index,
                                                               int cardinality) const {
    int size = static_cast<int>(tour.sizes().second);
    int split = preservedCardinality(cardinality, size, *ctx.random);
    int total = cardinality + split;
    auto bounds = lowerBounds(total, size, index) | ranges::to_vector;

    auto startTotal = bounds.at(ctx.random->uniform<int>(0, bounds.size() - 1));
    auto splitStart = ctx.random->uniform<int>(startTotal, startTotal + cardinality - 1);
    auto splitEnd = splitStart + split;

    return ranges::view::for_each(ranges::view::ints(startTotal, startTotal + total), [=, &tour](int i) {
      auto j = tour.get(static_cast<size_t>(i))->job;
      auto isSplit = i >= splitStart && i < splitEnd && i != index;
      return ranges::yield_if(!isSplit && j.has_value(), j.value());
    });
  }

  // endregion

  // region String utils

  /// Returns all possible lower bounds of the string.
  ranges::any_view<int> lowerBounds(int stringCardinality, int tourCardinality, int index) const {
    return ranges::view::for_each(ranges::view::closed_indices(1, stringCardinality), [=](const auto i) {
      int lower = index - (stringCardinality - i);
      int upper = index + (i - 1);
      return ranges::yield_if(lower >= 0 && upper < tourCardinality, lower);
    });
  }

  /// Calculates preserved substring cardinality.
  int preservedCardinality(int stringCardinality, int tourCardinality, utils::Random& random) const {
    if (stringCardinality == tourCardinality) return 0;

    int preservedCardinality = 1;
    while (stringCardinality + preservedCardinality < tourCardinality) {
      if (random.uniform<double>(0, 1) < alpha) {
        return preservedCardinality;
      } else
        ++preservedCardinality;
    }
    return preservedCardinality;
  }

  // endregion
};
}
