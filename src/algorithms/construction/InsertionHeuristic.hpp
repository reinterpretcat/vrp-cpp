#pragma once

#include "algorithms/construction/InsertionContext.hpp"

namespace vrp::algorithms::construction {

/// Specifies generic insertion heuristic interface.
template<typename Algorithm>
struct InsertionHeuristic {
  InsertionContext insert(const InsertionContext& ctx) const { return static_cast<Algorithm*>(this)->analyze(ctx); }
};
}
