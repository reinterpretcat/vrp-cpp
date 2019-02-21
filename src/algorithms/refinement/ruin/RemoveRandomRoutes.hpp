#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"

namespace vrp::algorithms::refinement {

/// Removes random routes from insertion context.
struct RemoveRandomRoutes final {
  /// Specifies minimum amount of removed routes.
  int rmin = 1;

  /// Specifies maximum amount of removed routes.
  int rmax = 3;

  construction::InsertionContext operator()(const RefinementContext& rCtx,
                                            const models::Solution& sln,
                                            construction::InsertionContext&& iCtx) const {
    auto toDelete = std::min(static_cast<size_t>(rCtx.random->uniform<int>(rmin, rmax)), iCtx.routes.size());
    ranges::for_each(ranges::view::iota(0, toDelete), [&](auto) {
      auto routeIndex = rCtx.random->uniform<int>(0, static_cast<int>(iCtx.routes.size()) - 1);

      auto rs = iCtx.routes.begin();
      std::advance(rs, routeIndex);

      if (rCtx.locked->empty())
        removeFullRoute(iCtx, rs);
      else
        removePartRoute(rCtx, iCtx, rs);
    });

    return std::move(iCtx);
  }

private:
  using Iterator =
    std::set<construction::InsertionRouteContext, construction::compare_insertion_route_contexts>::iterator;

  void removeFullRoute(construction::InsertionContext& iCtx, Iterator rs) const {
    ranges::copy(rs->route->tour.jobs(), ranges::inserter(iCtx.jobs, iCtx.jobs.begin()));

    iCtx.routes.erase(rs);
  }

  void removePartRoute(const RefinementContext& rCtx, construction::InsertionContext& iCtx, Iterator rs) const {
    using namespace ranges;

    auto toRemove = rs->route->tour.jobs() |
      view::remove_if([&](const auto& j) { return rCtx.locked->find(j) != rCtx.locked->end(); }) | to_vector;

    if (toRemove.size() == rs->route->tour.jobCount()) {
      removeFullRoute(iCtx, rs);
      return;
    }

    ranges::for_each(toRemove, [&](const auto& j) {
      rs->route->tour.remove(j);
      iCtx.jobs.push_back(j);
    });
    rCtx.problem->constraint->accept(const_cast<construction::InsertionRouteContext&>(*rs));
  }
};
}
