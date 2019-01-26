#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/evaluators/SequenceInsertionEvaluator.hpp"
#include "algorithms/construction/evaluators/ServiceInsertionEvaluator.hpp"
#include "algorithms/construction/extensions/Insertions.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"

#include <chrono>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {

  /// Evaluates possibility to preform insertion from given insertion context.
  InsertionResult evaluate(const models::problem::Job& job, const InsertionContext& ctx) const {
    using namespace ranges;
    using namespace models::solution;

    // iterate through list of routes plus a new one
    return ranges::accumulate(
      view::concat(ctx.routes, ctx.registry->next() | view::transform([&](const auto& a) {
                                 auto [start, end] = waypoints(*a);
                                 return InsertionRouteContext{std::make_shared<Route>(Route{a, start, end, {}}),
                                                              std::make_shared<InsertionRouteState>()};
                               })),
      make_result_failure(),
      [&](const auto& acc, const auto& routeCtx) {
        auto progress =
          build_insertion_progress{}
            .cost(acc.index() == 0 ? ranges::get<0>(acc).cost : std::numeric_limits<models::common::Cost>::max())
            .total(ctx.progress.total)
            .completeness(ctx.progress.completeness)
            .owned();

        // check hard constraints on route level.
        auto error = ctx.constraint->hard(routeCtx, job);
        if (error.has_value())
          return get_best_result(acc, {ranges::emplaced_index<1>, InsertionFailure{error.value()}});

        // evaluate its insertion cost
        auto result = models::problem::analyze_job<InsertionResult>(
          job,
          [&](const std::shared_ptr<const models::problem::Service>& service) {
            return ServiceInsertionEvaluator{}.evaluate(job, service, routeCtx, *ctx.constraint, progress);
          },
          [&](const std::shared_ptr<const models::problem::Sequence>& sequence) {
            return SequenceInsertionEvaluator{}.evaluate(job, sequence, routeCtx, *ctx.constraint, progress);
          });

        // propagate best result or failure
        return get_best_result(acc, result);
      });
  }

private:
  using Activity = models::solution::Tour::Activity;

  /// Creates start and end waypoints for given actor.
  std::pair<Activity, Activity> waypoints(const models::solution::Actor& actor) const {
    using namespace vrp::utils;
    using namespace vrp::models;

    const auto& detail = actor.detail;

    // create start/end for new vehicle
    auto start = solution::build_activity{}
                   .type(solution::Activity::Type::Start)
                   .detail({detail.start, 0, {detail.time.start, std::numeric_limits<common::Timestamp>::max()}})
                   .schedule({detail.time.start, detail.time.start})  //
                   .shared();
    auto end = solution::build_activity{}
                 .type(solution::Activity::Type::End)
                 .detail({detail.end.value_or(detail.start), 0, {0, detail.time.end}})
                 .schedule({detail.time.end, detail.time.end})  //
                 .shared();

    return {start, end};
  }
};

}  // namespace vrp::algorithms::construction