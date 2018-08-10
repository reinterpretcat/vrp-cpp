#ifndef VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP
#define VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP

#include "models/Solution.hpp"

namespace vrp {
namespace algorithms {
namespace costs {

/// Calculates total cost of solution.
struct calculate_total_cost final {
  vrp::models::Solution::Shadow solution;
  ANY_EXEC_UNIT float operator()(int index) const;
};

}  // namespace costs
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_COSTS_SOLUTIONCOSTS_HPP