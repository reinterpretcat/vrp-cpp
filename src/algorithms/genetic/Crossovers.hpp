#ifndef VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
#define VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP

#include "algorithms/genetic/Models.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Implements Adjusted Cost Difference Convolution crossover.
struct adjusted_cost_difference final {
  vrp::models::Convolutions operator()(const vrp::models::Problem& problem,
                                       vrp::models::Tasks& tasks,
                                       const Settings& settings,
                                       const Generation& generation) const;
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
