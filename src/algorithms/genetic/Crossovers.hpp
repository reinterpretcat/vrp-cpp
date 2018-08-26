#ifndef VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
#define VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP

#include "algorithms/genetic/Models.hpp"
#include "models/Solution.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Implements Adjusted Cost Difference Convolution crossover.
template<typename... Heuristics>
struct adjusted_cost_difference final {
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;

  EXEC_UNIT void operator()(const Generation& generation) const;
};

/// Do nothing.
struct empty_crossover final {
  ANY_EXEC_UNIT void operator()(const Generation& generation) const {}
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
