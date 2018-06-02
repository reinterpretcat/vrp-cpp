#ifndef VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
#define VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP

#include "algorithms/genetic/Models.hpp"
#include "models/Solution.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Implements Adjusted Cost Difference Convolution crossover.
struct adjusted_cost_difference final {
  vrp::models::Convolutions operator()(vrp::models::Solution& solution,
                                       const Settings& settings,
                                       const Generation& generation) const;
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
