#ifndef VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP
#define VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"

namespace vrp {
namespace algorithms {
namespace costs {

/// Calculates total cost of solution.
struct calculate_total_cost final {
  __host__ float operator()(const vrp::models::Problem& problem,
                            vrp::models::Tasks& tasks,
                            int solution = 0) const;
};

}  // namespace costs
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP