#ifndef VRP_ALGORITHMS_GENETIC_MODELS_HPP
#define VRP_ALGORITHMS_GENETIC_MODELS_HPP

#include "algorithms/convolutions/Models.hpp"
#include "models/Solution.hpp"

#include <thrust/random/linear_congruential_engine.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Encapsulates genetic algorithm settings.
struct Settings final {
  /// Size of population.
  int populationSize;
};

/// Specifies settings for next generation.
struct Generation {
  /// Parents indices.
  thrust::pair<int, int> parents;
  /// Offspring indices.
  thrust::pair<int, int> offspring;
  /// Convolution settings.
  vrp::algorithms::convolutions::Settings settings;
};

/// Defines parameters of mutation.
struct Mutation {
  /// Source individuum.
  int source;
  /// Destination individuum
  int destination;
  /// Convolution settings.
  vrp::algorithms::convolutions::Settings settings;
};

/// Specifies selection settings.
struct Selection final {
  /// Individuums belong to elite group.
  int elite;
  /// Amount of crossover operations with settings.
  thrust::pair<int, vrp::algorithms::convolutions::Settings> crossovers;
  /// Amount of mutations within settings.
  thrust::pair<int, vrp::algorithms::convolutions::Settings> mutations;
};

/// Defines evolution context.
struct EvolutionContext {
  /// Generation index.
  int generation;
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;
  /// Keeps costs of all individuums estimated for current generation.
  vrp::runtime::vector<thrust::pair<int, float>> costs;
  /// Random number generator.
  thrust::minstd_rand rng;
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_MODELS_HPP
