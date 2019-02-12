#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionRouteContext.hpp"
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
                  .prev(routeCtx->route->tour.start())
                  .target(target)
                  .next(routeCtx->route->tour.end())
                  .shared();
  return {routeCtx, actCtx};
}
}
