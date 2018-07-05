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

  /// Convolution settings
  vrp::algorithms::convolutions::Settings convolution;
};

/// Holds individuum indicies (solutions) to be processed.
struct Generation {
  thrust::pair<int, int> parents;
  thrust::pair<int, int> offspring;
};

/// Defines parameters of mutation.
struct Mutation {
  /// Affected individuum.
  int index;

  /// Defines how much individuum is affected, specified in range [0,1].
  float aggressiveness;

  /// Specifies whether convolutions are affected too.
  bool keepConvolutions;
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_MODELS_HPP
