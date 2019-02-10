#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/extensions/Insertions.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"
#include "utils/Collections.hpp"

#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {
private:
  /// Stores information needed for service insertion.
  struct SrvContext final {
    bool isStopped;                                      /// True, if processing has to be stopped.
    int code = 0;                                        /// Violation code.
    size_t index = 0;                                    /// Insertion index.
    models::common::Cost cost = models::common::NoCost;  /// Best cost.
    models::solution::Activity::Detail detail;           /// Activity detail.

    /// Creates a new context with index specified.
    static SrvContext empty(const models::common::Cost& cost = models::common::NoCost) {
      return {false, 0, 0, cost, {}};
    }

    /// Creates a new context from old one when insertion failed.
    static SrvContext fail(std::tuple<bool, int> error, const SrvContext& other) {
      return {std::get<0>(error), std::get<1>(error), other.index, other.cost, other.detail};
    }

    /// Creates a new context from old one when insertion worse.
    static SrvContext skip(const SrvContext& other) {
      return {other.isStopped, other.code, other.index, other.cost, other.detail};
    }

    /// Creates a new context.
    static SrvContext success(size_t index,
                              const models::common::Cost& cost,
                              const models::solution::Activity::Detail& detail) {
      return {false, 0, index, cost, detail};
    }

    /// Checks whether insertion is found.
    bool isSuccess() const { return cost < models::common::NoCost; }
  };

  /// Stores information needed for sequence insertion.
  struct SeqContext final {
    int code;                                                                     /// Violation code.
    size_t startIndex;                                                            /// Insertion index for first service.
    size_t index;                                                                 /// Insertion index for next service.
    std::optional<models::common::Cost> cost;                                     /// Cost accumulator.
    std::vector<std::pair<models::solution::Tour::Activity, size_t>> activities;  /// Activities with their indices

    static SeqContext&& forward(SeqContext&& left, SeqContext&& right) {
      auto index = std::max(left.startIndex, right.startIndex) + 1;
      left.startIndex = index;
      right.startIndex = index;
      left.index = index;
      right.index = index;
      return left.cost.has_value() && right.cost.has_value() ? std::move(left.cost < right.cost ? left : right)
                                                             : std::move(left.cost.has_value() ? left : right);
    }

    /// Creates empty context.
    static SeqContext empty() { return {0, 0, {}, {}}; }

    /// Creates failed insertion within reason code.
    static SeqContext fail(int code, size_t startIndex) {
      return {code, startIndex, startIndex, models::common::NoCost, {}};
    }

    /// Creates successful insertion context.
    static SeqContext success(models::common::Cost cost,
                              std::vector<std::pair<models::solution::Tour::Activity, size_t>>&& activities) {
      auto startIndex = activities.front().second;
      auto nextIndex = activities.back().second + 1;
      return {0, startIndex, nextIndex, cost, std::move(activities)};
    }

    /// Creates next insertion from existing one.
    SeqContext next() const { return SeqContext{0, startIndex, startIndex, 0, {}}; }

    /// Checks whether insertion is found.
    bool isSuccess() const { return code == 0 && cost.has_value(); }
  };

public:
  /// Evaluates possibility to preform insertion from given insertion context.
  InsertionResult evaluate(const models::problem::Job& job, const InsertionContext& ctx) const {
    using namespace ranges;
    using namespace models::solution;

    // iterate through list of routes plus a new one
    return ranges::accumulate(
      view::concat(ctx.routes, ctx.registry->next() | view::transform([&](const auto& a) {
                                 const auto& dtl = a->detail;
                                 auto start = build_activity{}
                                                .detail({dtl.start, 0, {dtl.time.start, models::common::MaxTime}})
                                                .schedule({dtl.time.start, dtl.time.start})
                                                .shared();
                                 auto end = build_activity{}
                                              .detail({dtl.end.value_or(dtl.start), 0, {0, dtl.time.end}})
                                              .schedule({dtl.time.end, dtl.time.end})
                                              .shared();
                                 return InsertionRouteContext{std::make_shared<Route>(Route{a, start, end, {}}),
                                                              std::make_shared<InsertionRouteState>()};
                               })),
      make_result_failure(),
      [&](const auto& acc, const auto& routeCtx) {
        // check hard constraints on route level.
        auto error = ctx.problem->constraint->hard(routeCtx, job);
        if (error.has_value())
          return get_best_result(acc, {ranges::emplaced_index<1>, InsertionFailure{error.value()}});

        // TODO we need just to change cost, simplify as this is performance expensive place
        auto progress =
          build_insertion_progress{}
            .cost(acc.index() == 0 ? ranges::get<0>(acc).cost : std::numeric_limits<models::common::Cost>::max())
            .total(ctx.progress.total)
            .completeness(ctx.progress.completeness)
            .owned();

        // evaluate its insertion cost
        auto result = models::problem::analyze_job<InsertionResult>(
          job,
          [&](const std::shared_ptr<const models::problem::Service>& service) {
            return evaluateService(job, service, ctx, routeCtx, progress);
          },
          [&](const std::shared_ptr<const models::problem::Sequence>& sequence) {
            return evaluateSequence(job, sequence, ctx, routeCtx, progress);
          });

        // propagate best result or failure
        return get_best_result(acc, result);
      });
  }

private:
  /// Evaluates service insertion.
  InsertionResult evaluateService(const models::problem::Job& job,
                                  const std::shared_ptr<const models::problem::Service>& service,
                                  const InsertionContext& iCtx,
                                  const InsertionRouteContext& rCtx,
                                  const InsertionProgress& progress) const {
    using namespace ranges;
    using namespace vrp::models;

    auto activity = std::make_shared<solution::Activity>(solution::Activity{{}, {}, service});

    const auto& constraint = *iCtx.problem->constraint;
    const auto& route = *rCtx.route;

    // calculate additional costs on route level.
    auto routeCosts = constraint.soft(rCtx, job);

    // form route legs from a new route view.
    auto tour = view::concat(view::single(route.start), route.tour.activities(), view::single(route.end));
    auto legs = view::zip(tour | view::sliding(2), view::iota(static_cast<size_t>(0)));
    auto evalCtx = SrvContext::empty(progress.bestCost);
    auto pred = [](const SrvContext& ctx) { return !ctx.isStopped; };

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
          auto status = constraint.hard(rCtx, actCtx);
          if (status.has_value()) return SrvContext::fail(status.value(), in2);

          auto totalCosts = routeCosts + constraint.soft(rCtx, actCtx);
          return totalCosts < in2.cost
            ? SrvContext::success(actCtx.index, totalCosts, {activity->detail.location, detail.duration, time})
            : SrvContext::skip(in2);
        });
      });
    });

    activity->detail = result.detail;

    return result.isSuccess()
      ? make_result_success({result.cost, as_job(activity->service.value()), {{activity, result.index}}, rCtx})
      : make_result_failure(result.code);
  }

  /// Evaluates sequence insertion.
  InsertionResult evaluateSequence(const models::problem::Job& job,
                                   const std::shared_ptr<const models::problem::Sequence>& sequence,
                                   const InsertionContext& iCtx,
                                   const InsertionRouteContext& rCtx,
                                   const InsertionProgress& progress) const {
    using namespace ranges;
    using namespace vrp::models;
    using namespace vrp::utils;
    using Activity = solution::Activity;

    static const auto srvPred = [](const SrvContext& acc) { return !acc.isStopped; };
    static const auto inSeqPred = [](const SeqContext& acc) { return acc.code == 0; };
    const auto outSeqPred = [=](const SeqContext& acc) { return acc.startIndex <= rCtx.route->tour.sizes().second; };

    // iterate through all possible insertion points
    auto result = accumulate_while(view::iota(0), SeqContext::empty(), outSeqPred, [&](auto& out, auto) {
      auto newCtx = deep_copy_insertion_route_context{}(rCtx);
      auto sqRes = accumulate_while(sequence->services, out.next(), inSeqPred, [&](auto& in1, const auto& service) {
        const auto& route = *newCtx.route;
        auto tour = view::concat(view::single(route.start), route.tour.activities(), view::single(route.end));
        auto legs = view::zip(tour | view::sliding(2), view::iota(static_cast<size_t>(0))) | view::drop(in1.index);
        auto activity = std::make_shared<Activity>(Activity{{}, {}, service});

        // NOTE condition below allows to stop at first success for first service to avoid situation
        // when later insertion of first service is cheaper, but the whole sequence is more expensive.
        // Due to complexity, we do this only for first service which is suboptimal.
        auto pred = [&](const SrvContext& acc) {
          return !(sequence->services.front() == service && acc.isSuccess()) && !acc.isStopped;
        };

        // region analyze legs and stop at best success or first failure
        auto srvRes = accumulate_while(legs, SrvContext::empty(), pred, [&](const auto& in2, const auto& leg) {
          auto [items, index] = leg;
          auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
          auto aCtx = InsertionActivityContext{index, prev, activity, next};

          // service details
          return accumulate_while(view::all(service->details), in2, srvPred, [&](const auto& in3, const auto& dtl) {
            // service time windows
            return accumulate_while(view::all(dtl.times), in3, srvPred, [&](const auto& in4, const auto& time) {
              aCtx.target->detail = {dtl.location.value_or(aCtx.prev->detail.location), dtl.duration, time};
              auto status = iCtx.problem->constraint->hard(newCtx, aCtx);
              if (status.has_value()) return SrvContext::fail(status.value(), in4);

              auto costs = iCtx.problem->constraint->soft(newCtx, aCtx);
              return costs < in4.cost
                ? SrvContext::success(aCtx.index, costs, {aCtx.target->detail.location, dtl.duration, time})
                : SrvContext::skip(in4);
            });
          });
        });
        // endregion

        if (srvRes.isSuccess()) {
          activity->detail = srvRes.detail;
          newCtx.route->tour.insert(activity, srvRes.index);
          iCtx.problem->constraint->accept(newCtx);
          return SeqContext::success(in1.cost.value() + srvRes.cost, concat(in1.activities, {activity, srvRes.index}));
        }

        return SeqContext::fail(srvRes.code, in1.startIndex);
      });

      return SeqContext::forward(std::move(sqRes), std::move(out));
    });

    return result.isSuccess() ? make_result_success({result.cost.value(), job, std::move(result.activities), rCtx})
                              : make_result_failure(result.code);
  }
};

}  // namespace vrp::algorithms::construction
