#ifndef VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
#define VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP

#include "algorithms/convolutions/Models.hpp"
#include "algorithms/genetic/Models.hpp"
#include "models/Solution.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Implements Adjusted Cost Difference Convolution crossover.
template<typename Heuristic>
struct adjusted_cost_difference final {
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;
  /// Object pool
  thrust::device_ptr<vrp::utils::DevicePool> pool;

  __device__ void operator()(const Settings& settings, const Generation& generation) const;
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
