#ifndef VRP_ALGORITHMS_GENETIC_TERMINATIONS_HPP
#define VRP_ALGORITHMS_GENETIC_TERMINATIONS_HPP

#include "algorithms/genetic/Models.hpp"


namespace vrp {
namespace algorithms {
namespace genetic {

/** Specifies criteria which limits evolution by max generations count. */
struct max_generations final {
  size_t max;

  EXEC_UNIT bool operator()(const EvolutionContext& context);
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_TERMINATIONS_HPP
