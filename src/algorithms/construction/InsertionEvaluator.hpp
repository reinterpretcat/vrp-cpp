#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/evaluators/ServiceInsertionEvaluator.hpp"
#include "algorithms/construction/evaluators/ShipmentInsertionEvaluator.hpp"
#include "algorithms/construction/extensions/Routes.hpp"
#include "algorithms/construction/extensions/States.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"
#include "utils/extensions/Variant.hpp"

#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Provides the way to evaluate insertion cost.
struct InsertionEvaluator final {
  InsertionEvaluator(const std::shared_ptr<const models::costs::TransportCosts>& transportCosts,
                     const std::shared_ptr<const models::costs::ActivityCosts>& activityCosts,
                     const std::shared_ptr<InsertionConstraint>& constraint) :
    serviceInsertionEvaluator(transportCosts, activityCosts, constraint),
    shipmentInsertionEvaluator(transportCosts, activityCosts, constraint) {}

  /// Evaluates possibility to preform insertion from given insertion context.
  InsertionResult evaluate(const models::problem::Job& job, const InsertionContext& ctx) const {
    using namespace ranges;
    using namespace models::solution;

    return ranges::accumulate(
      // create new route and use it within existing ones
      view::concat(view::single(InsertionContext::RouteState{std::make_shared<models::solution::Route>(),
                                                             std::make_shared<InsertionRouteState>()}),
                   view::all(ctx.routes | view::transform([](const auto& v) { return std::pair(v.first, v.second); }))),
      make_result_failure(),
      [&](const auto& outer, const auto& rs) {
        // determine current actor type hash
        auto type = rs.first->actor == nullptr ? 0 : actorHash(*rs.first->actor);
        // create list of all actors
        auto actors =
          view::concat(rs.first->actor == nullptr ? view::empty<Route::Actor>()
                                                  : static_cast<any_view<Route::Actor>>(view::single(rs.first->actor)),
                       ctx.registry->actors() | view::remove_if([=](const auto& a) { return actorHash(*a) == type; }));

        return ranges::accumulate(actors, outer, [&](const auto& inner, const auto& newActor) {
          // create actor specific route context
          auto routeCtx = createRouteContext(newActor, rs);

          // evaluate its insertion cost
          auto result = utils::mono_result<InsertionResult>(job.visit(ranges::overload(
            [&](const std::shared_ptr<const models::problem::Service>& service) {
              return serviceInsertionEvaluator.evaluate(service, routeCtx, ctx.progress);
            },
            [&](const std::shared_ptr<const models::problem::Shipment>& shipment) {
              return shipmentInsertionEvaluator.evaluate(shipment, routeCtx, ctx.progress);
            })));

          // propagate best result or failure
          return get_cheapest(inner, result);
        });
      });
  }

private:
  /// Creates new route context for given actor and route state.
  InsertionRouteContext createRouteContext(const models::solution::Route::Actor& actor,
                                           const InsertionContext::RouteState& routeState) const {
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