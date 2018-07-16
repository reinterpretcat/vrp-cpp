#ifndef VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
#define VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP

#include "algorithms/convolutions/Models.hpp"
#include "algorithms/genetic/Models.hpp"
#include "models/Solution.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Mutates individuum.
struct mutate_individuum final {
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;

  EXEC_UNIT void operator()(const Settings& settings, int index) const;
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_CROSSOVERS_HPP
