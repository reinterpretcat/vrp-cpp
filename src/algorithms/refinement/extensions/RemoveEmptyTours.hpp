#pragma once

#include "algorithms/construction/InsertionContext.hpp"

namespace vrp::algorithms::refinement {

/// Removes empty tours from insertion context.
struct remove_empty_tours final {
  void operator()(construction::InsertionContext& ctx) const {
    for (auto it = ctx.routes.begin(); it != ctx.routes.end();) {
      auto isEmpty = it->route->tour.empty();
      if (isEmpty) {
        ctx.registry->free(it->route->actor);
        it = ctx.routes.erase(it);
      } else
        ++it;
    }
  }
};
}