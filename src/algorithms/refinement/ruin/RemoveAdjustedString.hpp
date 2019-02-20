#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "algorithms/refinement/extensions/RemoveEmptyTours.hpp"
#include "algorithms/refinement/extensions/RestoreInsertionContext.hpp"
#include "algorithms/refinement/extensions/SelectRandomJob.hpp"
#include "models/Solution.hpp"
#include "models/extensions/solution/Helpers.hpp"
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
  construction::InsertionContext operator()(const RefinementContext& rCtx,
                                            const models::Solution& sln,
                                            construction::InsertionContext&& iCtx) const {
    using namespace ranges;

    auto jobs = std::make_shared<std::set<models::problem::Job, models::problem::compare_jobs>>();
    auto routes = std::make_shared<std::set<std::shared_ptr<models::solution::Route>>>();
    auto [lsmax, ks] = limits(rCtx, sln);

    ranges::for_each(
      view::take_while(
        selectJobs(rCtx, sln) | view::remove_if([&](const auto& j) { return in(*jobs, j) || in(sln.unassigned, j); }),
        [ks = ks, routes](const auto&) { return routes->size() != ks; }),
      [=, &rCtx, &iCtx, lsmax = lsmax](const auto& job) {
        ranges::for_each(
          iCtx.routes | view::remove_if([&](const auto& r) { return in(*routes, r.route) || !r.route->tour.has(job); }),
          [=, &rCtx](const auto& routeState) {
            /// Equations 8, 9: calculate cardinality of the string removed from the tour
            auto ltmax = std::min(static_cast<double>(routeState.route->tour.count()), lsmax);
            auto lt = static_cast<int>(std::floor(rCtx.random->uniform<double>(1, ltmax + 1)));

            auto toRemove = selectString(rCtx, routeState.route->tour, job, lt) |
              view::remove_if([&](const auto& j) { return in(*rCtx.locked, j); }) | to_vector;

            routes->insert(routeState.route);
            ranges::for_each(toRemove, [&](const auto& j) {
              routeState.route->tour.remove(j);
              jobs->insert(j);
            });
            rCtx.problem->constraint->accept(const_cast<construction::InsertionRouteContext&>(routeState));
          });
      });

    ranges::copy(*jobs, ranges::inserter(iCtx.jobs, iCtx.jobs.begin()));

    remove_empty_tours{}(iCtx);

    return std::move(iCtx);
  }

private:
  constexpr static int JobActivityStartIndex = 1;
  template<typename T, typename C>
  bool in(const std::set<T, C>& set, const T& item) const {
    return set.find(item) != set.end();
  }

  template<typename T, typename C>
  bool in(const std::map<T, int, C>& map, const T& item) const {
    return map.find(item) != map.end();
  }

  /// Calculates initial parameters from paper using 5,6,7 equations.
  std::tuple<double, int> limits(const RefinementContext& ctx, const models::Solution& sln) const {
    /// Equation 5: max removed string cardinality for each tour
    double lsmax = std::min(static_cast<double>(lmax), avgTourCardinality(sln));

    /// Equation 6: max number of strings
    double ksmax = 4 * cavg / (1 + lsmax) - 1;

    /// Equation 7: number of string to be removed
    int ks = static_cast<int>(std::floor(ctx.random->uniform<double>(1, ksmax + 1)));

    return {lsmax, ks};
  }

  /// Calculates average tour cardinality.
  double avgTourCardinality(const models::Solution& sln) const {
    return std::round(
      ranges::accumulate(sln.routes, 0.0, [](const double acc, const auto& r) { return acc + r->tour.count(); }) /
      sln.routes.size());
  }

  /// Returns randomly selected job and all its neighbours.
  ranges::any_view<models::problem::Job> selectJobs(const RefinementContext& ctx, const models::Solution& sln) const {
    auto seed = select_random_job{}(sln.routes, *ctx.random);
    if (!seed) return ranges::view::empty<models::problem::Job>();

    auto [route, job] = seed.value();

    return ranges::view::concat(
      ranges::view::single(job),
      ctx.problem->jobs->neighbors(route->actor->vehicle->profile, job, models::common::Timestamp{0}));
  }

  // region String selection

  /// Removes string for selected customer.
  ranges::any_view<models::problem::Job> selectString(const RefinementContext& ctx,
                                                      const models::solution::Tour& tour,
                                                      const models::problem::Job& job,
                                                      int cardinality) const {
    auto index = static_cast<int>(tour.index(job));
    return ctx.random->isHeadsNotTails() ? sequentialString(ctx, tour, index, cardinality)
                                         : preservedString(ctx, tour, index, cardinality);
  }

  /// Selects sequential string.
  ranges::any_view<models::problem::Job> sequentialString(const RefinementContext& ctx,
                                                          const models::solution::Tour& tour,
                                                          int index,
                                                          int cardinality) const {
    using namespace ranges;

    int size = static_cast<int>(tour.count());

    auto [begin, end] = lowerBounds(cardinality, size, index);
    auto start = ctx.random->uniform<int>(begin, end);

    return view::for_each(view::ints(start, start + cardinality) | view::reverse, [&tour](int i) {
      auto j = models::solution::retrieve_job{}(*tour.get(static_cast<size_t>(i)));
      return ranges::yield_if(j.has_value(), j.value());
    });
  }

  /// Selects string with preserved customers.
  ranges::any_view<models::problem::Job> preservedString(const RefinementContext& ctx,
                                                         const models::solution::Tour& tour,
                                                         int index,
                                                         int cardinality) const {
    using namespace ranges;

    int size = static_cast<int>(tour.count());

    int split = preservedCardinality(cardinality, size, *ctx.random);
    int total = cardinality + split;

    auto [begin, end] = lowerBounds(total, size, index);
    auto startTotal = ctx.random->uniform<int>(begin, end);

    auto splitStart = ctx.random->uniform<int>(startTotal, startTotal + cardinality - 1);
    auto splitEnd = splitStart + split;

    // NOTE if selected job is in split range we should remove it anyway,
    // this line makes sure that string cardinality is kept as requested.
    total -= (index >= splitStart && index < splitEnd ? 1 : 0);

    return view::for_each(view::ints(startTotal, startTotal + total) | view::reverse, [=, &tour](int i) {
      auto j = models::solution::retrieve_job{}(*tour.get(static_cast<size_t>(i)));
      auto isSplit = i >= splitStart && i < splitEnd && i != index;
      return ranges::yield_if(!isSplit && j.has_value(), j.value());
    });
  }

  // endregion

  // region String utils

  /// Returns range of possible lower bounds.
  std::pair<int, int> lowerBounds(int stringCardinality, int tourCardinality, int index) const {
    auto start = std::max(JobActivityStartIndex, index - stringCardinality + 1);
    auto end = std::min(tourCardinality - stringCardinality + 1, start + stringCardinality);

    return {start, end};
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
