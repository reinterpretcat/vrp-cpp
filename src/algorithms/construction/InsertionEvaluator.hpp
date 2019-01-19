#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/evaluators/ServiceInsertionEvaluator.hpp"
#include "algorithms/construction/evaluators/ShipmentInsertionEvaluator.hpp"
#include "algorithms/construction/extensions/Insertions.hpp"
#include "algorithms/construction/extensions/Routes.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"
#include "utils/extensions/Variant.hpp"

#include <numeric>
#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {
  InsertionEvaluator(const std::shared_ptr<const models::costs::TransportCosts>& transportCosts,
                     const std::shared_ptr<const models::costs::ActivityCosts>& activityCosts) :
    serviceInsertionEvaluator(transportCosts, activityCosts),
    shipmentInsertionEvaluator(transportCosts, activityCosts) {}

  /// Evaluates possibility to preform insertion from given insertion context.
  InsertionResult evaluate(const models::problem::Job& job, const InsertionContext& ctx) const {
    using namespace ranges;
    using namespace models::solution;

    return ranges::accumulate(
      // iterate through list of routes plus a new one
      view::concat(view::all(ctx.routes | view::transform([](const auto& v) { return std::pair(v.first, v.second); })),
                   view::single(InsertionRouteContext::RouteState{std::make_shared<models::solution::Route>(),
                                                                  std::make_shared<InsertionRouteState>()})),
      make_result_failure(),
      [&](const auto& outer, const auto& rs) {
        // create list of all actors
        auto actors = view::concat(
          rs.first->actor == nullptr ? view::empty<Route::Actor>()
                                     : static_cast<any_view<Route::Actor>>(view::single(rs.first->actor)),
          ctx.registry->unique() | ranges::view::remove_if([&](const auto& a) { return a == rs.first->actor; }));

        return ranges::accumulate(actors, outer, [&](const auto& inner, const auto& newActor) {
          // create actor specific route context
          auto routeCtx = createRouteContext(newActor, rs);
          // create a new progress to reflect best at the moment known cost of given insertion
          auto progress =
            build_insertion_progress{}
              .cost(utils::mono_result<models::common::Cost>(inner.visit(ranges::overload(
                [](const InsertionSuccess& success) { return success.cost; },
                [](const InsertionFailure& failure) { return std::numeric_limits<models::common::Cost>::max(); }))))
              .total(ctx.progress.total)
              .completeness(ctx.progress.completeness)
              .owned();

          // evaluate its insertion cost
          auto result = utils::mono_result<InsertionResult>(job.visit(ranges::overload(
            [&](const std::shared_ptr<const models::problem::Service>& service) {
              return serviceInsertionEvaluator.evaluate(service, routeCtx, *ctx.constraint, progress);
            },
            [&](const std::shared_ptr<const models::problem::Shipment>& shipment) {
              return shipmentInsertionEvaluator.evaluate(shipment, routeCtx, *ctx.constraint, progress);
            })));

          // propagate best result or failure
          return get_best_result(inner, result);
        });
      });
  }

private:
  /// Creates new route context for given actor and route state.
  InsertionRouteContext createRouteContext(const models::solution::Route::Actor& actor,
                                           const InsertionRouteContext::RouteState& routeState) const {
    auto ctx = InsertionRouteContext{routeState, actor, 0};

    // route is used first time
    if (ctx.route.first->actor == nullptr) {
      auto [start, end] = waypoints(*actor, actor->detail.time.start);
      ctx.route.first->start = start;
      ctx.route.first->end = end;
      ctx.route.first->actor = actor;
      ctx.departure = actor->detail.time.start;
    } else {
      ctx.departure = routeState.first->start->schedule.departure;
    }

    return std::move(ctx);
  }

  const ServiceInsertionEvaluator serviceInsertionEvaluator;
  const ShipmentInsertionEvaluator shipmentInsertionEvaluator;
};

}  // namespace vrp::algorithms::construction