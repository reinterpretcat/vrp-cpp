#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/extensions/Insertions.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"

#include <numeric>
#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {
private:
  /// Defines evaluation context.
  struct Context final {
    bool isStopped;    /// True, if processing has to be stopped.
    int code = 0;      /// Violation code.
    size_t index = 0;  /// Insertion index.
    models::common::Cost cost = std::numeric_limits<models::common::Cost>::max();  /// Best cost.
    models::solution::Activity::Detail detail;                                     /// Activity detail.

    /// Creates a new context.
    static Context empty(const models::common::Cost& cost) { return {false, 0, 0, cost, {}}; }

    /// Creates a new context from old one when insertion failed.
    static Context fail(std::tuple<bool, int> error, const Context& other) {
      return {std::get<0>(error), std::get<1>(error), other.index, other.cost, other.detail};
    }

    /// Creates a new context from old one when insertion worse.
    static Context skip(const Context& other) {
      return {other.isStopped, other.code, other.index, other.cost, other.detail};
    }

    /// Creates a new context.
    static Context success(size_t index,
                           const models::common::Cost& cost,
                           const models::solution::Activity::Detail& detail) {
      return {false, 0, index, cost, detail};
    }

    /// Checks whether insertion is found.
    bool isSuccess() const { return cost < std::numeric_limits<models::common::Cost>::max(); }
  };

public:
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
            return evaluateService(job, service, routeCtx, *ctx.constraint, progress);
          },
          [&](const std::shared_ptr<const models::problem::Sequence>& sequence) {
            return evaluateSequence(job, sequence, routeCtx, *ctx.constraint, progress);
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

  /// Evaluates service insertion.
  InsertionResult evaluateService(const models::problem::Job& job,
                                  const std::shared_ptr<const models::problem::Service>& service,
                                  const InsertionRouteContext& ctx,
                                  const InsertionConstraint& constraint,
                                  const InsertionProgress& progress) const {
    using namespace ranges;
    using namespace vrp::models;
    using ActivityType = solution::Activity::Type;

    auto activity = std::make_shared<solution::Activity>(solution::Activity{ActivityType::Job, {}, {}, job});
    const auto& route = *ctx.route;

    // calculate additional costs on route level.
    auto routeCosts = constraint.soft(ctx, job);

    // form route legs from a new route view.
    auto tour = view::concat(view::single(route.start), route.tour.activities(), view::single(route.end));
    auto legs = view::zip(tour | view::sliding(2), view::iota(static_cast<size_t>(0)));
    auto evalCtx = Context::empty(progress.bestCost);
    auto pred = [](const Context& ctx) { return !ctx.isStopped; };

    // 1. analyze route legs
    auto result = utils::accumulate_while(legs, evalCtx, pred, [&](const auto& out, const auto& view) {
      auto [items, index] = view;
      auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
      auto actCtx = InsertionActivityContext{index, prev, activity, next};

      // 2. analyze service details
      return utils::accumulate_while(view::all(service->details), out, pred, [&](const auto& in1, const auto& detail) {
        // TODO check whether tw is empty
        // 3. analyze detail time windows
        return utils::accumulate_while(view::all(detail.times), in1, pred, [&](const auto& in2, const auto& time) {
          activity->detail = {detail.location.value_or(actCtx.prev->detail.location), detail.duration, time};
          // check hard activity constraint
          auto status = constraint.hard(ctx, actCtx);
          if (status.has_value()) return Context::fail(status.value(), in2);

          auto totalCosts = routeCosts + constraint.soft(ctx, actCtx);
          return totalCosts < in2.cost
            ? Context::success(actCtx.index, totalCosts, {activity->detail.location, detail.duration, time})
            : Context::skip(in2);
        });
      });
    });

    activity->detail = result.detail;

    return result.isSuccess()
      ? make_result_success({result.cost, activity->job.value(), {{activity, result.index}}, ctx})
      : make_result_failure(result.code);
  }

  /// Evaluates service insertion.
  InsertionResult evaluateSequence(const models::problem::Job& job,
                                   const std::shared_ptr<const models::problem::Sequence>& sequence,
                                   const InsertionRouteContext& ctx,
                                   const InsertionConstraint& constraint,
                                   const InsertionProgress& progress) const {
    // TODO
    return make_result_failure(0);
  }
};

}  // namespace vrp::algorithms::construction