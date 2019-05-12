#pragma once

#include "algorithms/refinement/RefinementContext.hpp"

#include <gsl/gsl>

namespace vrp::algorithms::refinement {

/// Selects the best solution from population.
struct select_best_solution final {
  models::EstimatedSolution operator()(const RefinementContext& ctx) const {
    Expects(!ctx.population->empty());
    // NOTE assume that population is sorted in ascending order.
    return ctx.population->front();
  }
};
}