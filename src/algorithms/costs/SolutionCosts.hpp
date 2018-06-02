#ifndef VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP
#define VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP

#include "models/Solution.hpp"

namespace vrp {
namespace algorithms {
namespace costs {

/// Calculates total cost of solution.
struct calculate_total_cost final {
  __host__ float operator()(vrp::models::Solution& solution, int index = 0) const;
};

}  // namespace costs
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP