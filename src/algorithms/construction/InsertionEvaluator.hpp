#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/extensions/Factories.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"
#include "utils/Collections.hpp"

#include <numeric>
#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {
  using PermutationFunc = std::function<std::vector<std::vector<int>>(const models::problem::Sequence&)>;

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
    static SrvContext fail(const std::tuple<bool, int>& error, const SrvContext& other) {
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
    bool isStopped;                                                               /// True if processing stopped.
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
    static SeqContext empty() { return {false, 0, 0, 0, {}, {}}; }

    /// Creates failed insertion within reason code.
    static SeqContext fail(const SrvContext& errCtx, const SeqContext& other) {
      auto isStopped = errCtx.isStopped && other.activities.empty();
      return {isStopped, errCtx.code, other.startIndex, other.startIndex, models::common::NoCost, {}};
    }

    /// Creates successful insertion context.
    static SeqContext success(models::common::Cost cost,
                              std::vector<std::pair<models::solution::Tour::Activity, size_t>>&& activities) {
      auto startIndex = activities.front().second;
      auto nextIndex = activities.back().second + 1;
      return {false, 0, startIndex, nextIndex, cost, std::move(activities)};
    }

    /// Creates next insertion from existing one.
    SeqContext next() const { return SeqContext{false, 0, startIndex, startIndex, 0, {}}; }

    /// Checks whether insertion is found.
    bool isSuccess() const { return code == 0 && cost.has_value(); }
  };

  /// Provides the way to use copy on write strategy within route state context.
  struct ShadowContext final {
    bool mutated;
    bool dirty;
    std::shared_ptr<const models::Problem> problem;
    InsertionRouteContext ctx;

    void insert(models::solution::Tour::Activity& activity, size_t index) {
      if (!mutated) {
        ctx = deep_copy_insertion_route_context{}(ctx);
        mutated = true;
      }

      ctx.route->tour.insert(activity, index + 1);
      problem->constraint->accept(ctx);
      dirty = true;
    }

    void restore(const models::problem::Job& job) {
      if (dirty) {
        ranges::for_each(ctx.route->tour.activities(job), [&](const auto& a) { ctx.state->remove(a); });
        ctx.route->tour.remove(job);
        problem->constraint->accept(ctx);
      }
      dirty = false;
    }
  };

  /// Retrieves permutations from sequence's dimens.
  /// TODO make it memory efficient
  struct retrieve_permutations final {
    using ServicePermutations = std::vector<std::vector<std::shared_ptr<const models::problem::Service>>>;

    ServicePermutations operator()(const models::problem::Sequence& sequence) const {
      using namespace ranges;

      auto permFunc = sequence.dimens.find(models::problem::Sequence::PermutationDimKey);

      if (permFunc == sequence.dimens.end()) return ranges::view::single(ranges::view::all(sequence.services));

      auto permutations = std::any_cast<std::shared_ptr<PermutationFunc>>(permFunc->second)->operator()(sequence);
      return permutations | view::transform([&](const auto& permutation) {
               return view::transform(permutation, [&](const auto index) { return sequence.services.at(index); }) |
                 to_vector;
             }) |
        to_vector;
    }
  };

public:
  /// Evaluates possibility to preform insertion from given insertion context.
  InsertionResult evaluate(const models::problem::Job& job, const InsertionContext& ctx) const {
    using namespace ranges;
    using namespace models::solution;

    // iterate through list of routes plus new ones
    return ranges::accumulate(
      view::concat(ctx.solution->routes, ctx.solution->registry->next() | view::transform([&](const auto& a) {
                                           return create_insertion_route_context{}(a);
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
    using namespace vrp::utils;

    auto activity = std::make_shared<solution::Activity>(solution::Activity{{}, {}, service});

    const auto& constraint = *iCtx.problem->constraint;
    const auto& route = *rCtx.route;

    // calculate additional costs on route level.
    auto routeCosts = constraint.soft(rCtx, job);

    auto evalCtx = SrvContext::empty(progress.bestCost);
    auto pred = [](const SrvContext& ctx) { return !ctx.isStopped; };

    // 1. analyze route legs
    auto result = accumulate_while(route.tour.legs(), std::move(evalCtx), pred, [&](auto& out, const auto& view) {
      using NextAct = std::optional<models::solution::Tour::Activity>;

      auto [items, index] = view;
      auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
      auto actCtx = InsertionActivityContext{index, prev, activity, prev == next ? NextAct{} : NextAct{next}};

      // 2. analyze service details
      return accumulate_while(view::all(service->details), std::move(out), pred, [&](auto& in1, const auto& detail) {
        // TODO check whether tw is empty
        // 3. analyze detail time windows
        return accumulate_while(view::all(detail.times), std::move(in1), pred, [&](auto& in2, const auto& time) {
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

    static const auto srvPred = [](const SrvContext& acc) { return !acc.isStopped; };
    static const auto inSeqPred = [](const SeqContext& acc) { return acc.code == 0; };
    const auto outSeqPred = [=](const SeqContext& acc) {
      return !acc.isStopped && acc.startIndex <= rCtx.route->tour.count();
    };

    auto result = ranges::accumulate(
      retrieve_permutations{}(*sequence), SeqContext::empty(), [&](auto& accRes, const auto& services) {
        auto shadow = ShadowContext{false, false, iCtx.problem, rCtx};
        auto permRes = accumulate_while(view::iota(0), SeqContext::empty(), outSeqPred, [&](auto& out, const auto) {
          shadow.restore(job);
          auto sqRes = accumulate_while(services, out.next(), inSeqPred, [&](auto& in1, const auto& service) {
            const auto& route = *shadow.ctx.route;
            auto activity = std::make_shared<solution::Activity>(solution::Activity{{}, {}, service});
            auto legs = route.tour.legs() | view::drop(in1.index);
            // NOTE condition below allows to stop at first success for first service to avoid situation
            // when later insertion of first service is cheaper, but the whole sequence is more expensive.
            // Due to complexity, we do this only for first service which is suboptimal for more than two services.
            auto pred = [&](const SrvContext& acc) {
              return !(sequence->services.front() == service && acc.isSuccess()) && !acc.isStopped;
            };

            // region analyze legs
            auto srvRes = accumulate_while(legs, SrvContext::empty(), pred, [&](auto& in2, const auto& leg) {
              using NextAct = std::optional<models::solution::Tour::Activity>;

              auto [items, index] = leg;
              auto [prev, next] = std::tie(*std::begin(items), *(std::begin(items) + 1));
              auto aCtx = InsertionActivityContext{index, prev, activity, prev == next ? NextAct{} : NextAct{next}};

              // service details
              return accumulate_while(service->details, std::move(in2), srvPred, [&](auto& in3, const auto& dtl) {
                // service time windows
                return accumulate_while(dtl.times, std::move(in3), srvPred, [&](auto& in4, const auto& time) {
                  aCtx.target->detail = {dtl.location.value_or(aCtx.prev->detail.location), dtl.duration, time};
                  auto status = iCtx.problem->constraint->hard(shadow.ctx, aCtx);
                  if (status.has_value()) return SrvContext::fail(status.value(), in4);

                  auto costs = iCtx.problem->constraint->soft(shadow.ctx, aCtx);
                  return costs < in4.cost
                    ? SrvContext::success(aCtx.index, costs, {aCtx.target->detail.location, dtl.duration, time})
                    : SrvContext::skip(in4);
                });
              });
            });
            // endregion

            if (srvRes.isSuccess()) {
              activity->detail = srvRes.detail;
              shadow.insert(activity, srvRes.index);
              return SeqContext::success(in1.cost.value() + srvRes.cost,
                                         concat(in1.activities, {activity, srvRes.index}));
            }

            return SeqContext::fail(srvRes, in1);
          });

          return SeqContext::forward(std::move(sqRes), std::move(out));
        });

        return SeqContext::forward(std::move(permRes), std::move(accRes));
      });

    return result.isSuccess() ? make_result_success({result.cost.value(), job, std::move(result.activities), rCtx})
                              : make_result_failure(result.code);
  }
};

}  // namespace vrp::algorithms::construction
