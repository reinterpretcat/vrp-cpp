#ifndef VRP_ALGORITHMS_GENETIC_EVOLUTION_HPP
#define VRP_ALGORITHMS_GENETIC_EVOLUTION_HPP

#include "algorithms/genetic/Models.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"

namespace vrp {
namespace algorithms {
namespace genetic {

/** Runs evolutions and terminates based on termination criteria. */
template<typename Strategy>
struct run_evolution final {
  Strategy strategy;

  vrp::models::Solution operator()(vrp::models::Problem&& problem);
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_EVOLUTION_HPP
