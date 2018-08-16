#ifndef VRP_ALGORITHMS_GENETIC_STRATEGIES_HPP
#define VRP_ALGORITHMS_GENETIC_STRATEGIES_HPP

#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/genetic/Models.hpp"
#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"

#include "models/Problem.hpp"
#include "models/Tasks.hpp"

namespace vrp {
namespace algorithms {
namespace genetic {

class LinearStrategy {
 public:
  typedef adjusted_cost_difference<
    vrp::algorithms::heuristics::nearest_neighbor<vrp::algorithms::heuristics::TransitionOperator>>
    Crossover;

  typedef create_mutant<vrp::algorithms::heuristics::TransitionOperator> Mutator;

  const Settings settings;

  /// Creates initial population
  vrp::models::Tasks population(const vrp::models::Problem& problem);

  /// Creates crossover.
  Crossover crossover(const EvolutionContext& ctx);

  /// Creates mutator.
  Mutator mutator(const EvolutionContext& ctx);

  /// Creates selection settings.
  Selection selection(const EvolutionContext& ctx);

  /// Loops to next generation.
  bool next(EvolutionContext& ctx);
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_STRATEGIES_HPP
