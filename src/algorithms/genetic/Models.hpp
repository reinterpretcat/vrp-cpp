#ifndef VRP_ALGORITHMS_GENETIC_MODELS_HPP
#define VRP_ALGORITHMS_GENETIC_MODELS_HPP

#include "algorithms/convolutions/Models.hpp"

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

/// Defines evolution context.
struct EvolutionContext {
  /// Generation index.
  int generation;
  /// Best known cost.
  float cost;
  /// Best known solution.
  int solution;
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_MODELS_HPP
