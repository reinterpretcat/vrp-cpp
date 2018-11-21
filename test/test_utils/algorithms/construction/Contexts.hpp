#pragma once

#include "algorithms/construction/insertion/InsertionActivityContext.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"

#include <memory>
#include <utility>

namespace vrp::test {

/// Encapsulates insertion contexts.
using TestInsertionContext = std::pair<std::shared_ptr<algorithms::construction::InsertionRouteContext>,
                                       std::shared_ptr<algorithms::construction::InsertionActivityContext>>;

/// Creates insertion contexts with the same actor.
inline TestInsertionContext
sameActor(const vrp::models::solution::Tour::Activity& prev,
          const vrp::models::solution::Tour::Activity& target,
          const vrp::models::solution::Tour::Activity& next) {
  auto routeCtx = test_build_insertion_route_context{}.shared();
  auto actCtx = test_build_insertion_activity_context{}  //
                  .prev(prev)
                  .target(target)
                  .next(next)
                  .shared();
  return {routeCtx, actCtx};
}

/// Creates insertion contexts with the same actor.
inline TestInsertionContext
sameActor(const vrp::models::solution::Tour::Activity& target) {
  auto routeCtx = test_build_insertion_route_context{}.shared();
  auto actCtx = test_build_insertion_activity_context{}  //
                  .prev(routeCtx->route->start)
                  .target(target)
                  .next(routeCtx->route->end)
                  .shared();
  return {routeCtx, actCtx};
}

/// Creates insertion contexts with different actor.
inline TestInsertionContext
differentActor(const vrp::models::solution::Tour::Activity& activity) {
  auto routeCtx =
    test_build_insertion_route_context{}
      .actor(test_build_actor{}.vehicle(test_build_vehicle{}.details({{20, {}, DefaultTimeWindow}}).shared()).shared())
      .shared();
  auto actCtx = test_build_insertion_activity_context{}  //
                  .prev(routeCtx->route->start)
                  .target(activity)
                  .next(routeCtx->route->end)
                  .shared();
  return {routeCtx, actCtx};
}

/// Creates insertion contexts with different actor.
inline TestInsertionContext
differentActor(const vrp::models::solution::Tour::Activity& prev,
               const vrp::models::solution::Tour::Activity& target,
               const vrp::models::solution::Tour::Activity& next,
               int returnLocation = -1) {
  auto actor = test_build_actor{}.time(DefaultTimeWindow).start(20);
  if (returnLocation > 0) {
    auto end = static_cast<models::common::Location>(returnLocation);
    actor.end(end).vehicle(test_build_vehicle{}.details({{20, std::make_optional(end), DefaultTimeWindow}}).shared());
  } else {
    actor.vehicle(test_build_vehicle{}.details({{20, {}, DefaultTimeWindow}}).shared());
  }

  auto routeCtx = test_build_insertion_route_context{}.actor(actor.shared()).shared();
  auto actCtx = test_build_insertion_activity_context{}  //
                  .prev(prev)
                  .target(target)
                  .next(next)
                  .shared();
  return {routeCtx, actCtx};
}
}
