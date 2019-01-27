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
  struct EvaluationContext final {
    bool isStopped;                                      /// True, if processing has to be stopped.
    int code = 0;                                        /// Violation code.
    size_t index = 0;                                    /// Insertion index.
    models::common::Cost cost = models::common::NoCost;  /// Best cost.
    models::solution::Activity::Detail detail;           /// Activity detail.

    /// Creates a new context with index specified.
    static EvaluationContext emptyWithCost(const models::common::Cost& cost) { return {false, 0, 0, cost, {}}; }

    /// Creates a new context with index specified.
    static EvaluationContext emptyWithIndex(size_t index) { return {false, 0, index, models::common::NoCost, {}}; }

    /// Creates a new context from old one when insertion failed.
    static EvaluationContext fail(std::tuple<bool, int> error, const EvaluationContext& other) {
      return {std::get<0>(error), std::get<1>(error), other.index, other.cost, other.detail};
    }

    /// Creates a new context from old one when insertion worse.
    static EvaluationContext skip(const EvaluationContext& other) {
      return {other.isStopped, other.code, other.index, other.cost, other.detail};
    }

    /// Creates a new context.
    static EvaluationContext success(size_t index,
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
            auto [result, activity] = evaluateService(job, *service, routeCtx, *ctx.constraint, progress);
            return result.isSuccess()
              ? make_result_success({result.cost, activity->job.value(), {{activity, result.index}}, routeCtx})
              : make_result_failure(result.code);
          },
          [&](const std::shared_ptr<const models::problem::Sequence>& sequence) {
            return evaluateSequence(job, *sequence, routeCtx, *ctx.constraint, progress);
          });

        // propagate best result or failure
        return get_best_result(acc, result);
      });
  }

private:
  using Activity = models::solution::Tour::Activity;

  /// Evaluates service insertion.
  std::pair<EvaluationContext, Activity> evaluateService(const models::problem::Job& job,
                                                         const models::problem::Service& service,
                                                         const InsertionRouteContext& ctx,
                                                         const InsertionConstraint& constraint,
                                                         const InsertionProgress& progress,
                                                         size_t startIndex = 0) const {
    using namespace ranges;
    using namespace vrp::models;
    using ActivityType = solution::Activity::Type;

    auto activity = std::make_shared<solution::Activity>(solution::Activity{ActivityType::Job, {}, {}, job});
    const auto& route = *ctx.route;

    // calculate additional costs on route level.
    auto routeCosts = constraint.soft(ctx, job);

    // form route legs from a new route view.
    auto tour = view::concat(view::single(route.start), route.tour.activities(), view::single(route.end));
    auto legs = view::zip(tour | view::drop(startIndex) | view::sliding(2), view::iota(static_cast<size_t>(0)));
    auto evalCtx = EvaluationContext::emptyWithCost(progress.bestCost);
    auto pred = [](const EvaluationContext& ctx) { return !ctx.isStopped; };

    // 1. analyze route legs
    auto result = utils::accumulate_while(legs, evalCtx, pred, [&](const auto& out, const auto& view) {
      auto [items, index] = view;
      auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
      auto actCtx = InsertionActivityContext{index, prev, activity, next};

      // 2. analyze service details
      return utils::accumulate_while(view::all(service.details), out, pred, [&](const auto& in1, const auto& detail) {
        // TODO check whether tw is empty
        // 3. analyze detail time windows
        return utils::accumulate_while(view::all(detail.times), in1, pred, [&](const auto& in2, const auto& time) {
          activity->detail = {detail.location.value_or(actCtx.prev->detail.location), detail.duration, time};
          // check hard activity constraint
          auto status = constraint.hard(ctx, actCtx);
          if (status.has_value()) return EvaluationContext::fail(status.value(), in2);

          auto totalCosts = routeCosts + constraint.soft(ctx, actCtx);
          return totalCosts < in2.cost
            ? EvaluationContext::success(actCtx.index, totalCosts, {activity->detail.location, detail.duration, time})
            : EvaluationContext::skip(in2);
        });
      });
    });

    activity->detail = result.detail;

    return {result, activity};
  }

  /// Evaluates sequence insertion.
  InsertionResult evaluateSequence(const models::problem::Job& job,
                                   const models::problem::Sequence& sequence,
                                   const InsertionRouteContext& routeCtx,
                                   const InsertionConstraint& constraint,
                                   const InsertionProgress& progress) const {
    using namespace ranges;
    using namespace vrp::models;
    using namespace vrp::utils;
    // <error code, next start index, best insertion cost>
    using Context = std::tuple<int, size_t, InsertionSuccess>;

    // 1. try different starting insertion points
    auto result = accumulate_while(  //
      view::ints(0),
      std::tuple<int, size_t, InsertionSuccess>{0, 0, InsertionSuccess{{}, job, {}, routeCtx}},
      [](const auto& r) {
        return std::get<0>(r) == 0;
        },
      [&](auto& ctx, auto) {
        auto success = InsertionSuccess{{}, job, {}, routeCtx};
        // 2. try to insert all services from sequence in tour starting from specific index
        auto result = accumulate_while(  //
          sequence.jobs,
          EvaluationContext::emptyWithIndex(std::get<1>(ctx)),
          [](const EvaluationContext& ec) { return !ec.isSuccess(); },
          [&](const auto& evalCtx, const auto& service) {
            auto [sResult, activity] = evaluateService(job, service, routeCtx, constraint, progress, evalCtx.index);

            if (sResult.isSuccess()) {
              success.cost += sResult.cost;
              success.activities.push_back({activity, sResult.index});
            }

            return sResult;
          });

        // TODO should be used
        bool hasMore = result.isSuccess() && !success.activities.empty() &&  //
          success.activities.front().second < routeCtx.route->tour.sizes().second;

        return result.isSuccess() && success.cost < std::get<2>(ctx).cost
          ? Context{result.code, success.activities.front().second, success}
          : Context{result.code, std::get<2>(ctx).cost, std::move(std::get<2>(ctx))};
      });

    return std::get<0>(result) > 0 ? InsertionResult{ranges::emplaced_index<1>, InsertionFailure{std::get<0>(result)}}
                                   : InsertionResult{ranges::emplaced_index<0>, std::move(std::get<2>(result))};
  }


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