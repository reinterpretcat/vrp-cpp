#ifndef VRP_SOLVER_GENETIC_CROSSOVERS_ADJUSTED_COST_DIFFERENCE_HPP
#define VRP_SOLVER_GENETIC_CROSSOVERS_ADJUSTED_COST_DIFFERENCE_HPP

#include "algorithms/genetic/Settings.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "utils/Pool.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Implements Adjusted Cost Difference Convolution crossover.
struct adjusted_cost_difference final {
  /// Holds individuum indicies to be processed.
  struct Generation {
    thrust::pair<int, int> parents;
    thrust::pair<int, int> offspring;
  };

  void operator()(const vrp::models::Problem& problem,
                  vrp::models::Tasks& tasks,
                  const Settings& settings,
                  const Generation& generation,
                  vrp::utils::Pool& pool) const;
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_SOLVER_GENETIC_CROSSOVERS_ADJUSTED_COST_DIFFERENCE_HPP
