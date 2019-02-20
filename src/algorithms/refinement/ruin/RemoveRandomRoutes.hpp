#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/refinement/RefinementContext.hpp"

namespace vrp::algorithms::refinement {

struct remove_random_routes final {
  construction::InsertionContext operator()(const RefinementContext& rCtx,
                                            construction::InsertionContext&& iCtx) const {
    return std::move(iCtx);
  }
};
}
