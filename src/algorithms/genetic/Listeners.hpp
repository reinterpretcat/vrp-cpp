#ifndef VRP_ALGORITHMS_GENETIC_LISTENERS_HPP
#define VRP_ALGORITHMS_GENETIC_LISTENERS_HPP

#include "algorithms/genetic/Models.hpp"
#include "models/Solution.hpp"

namespace vrp {
namespace algorithms {
namespace genetic {

/** Empty listener. */
struct empty_listener final {
  inline void operator()(const EvolutionContext& context) {}
};

/** Listener which tracks evolution for each generation. */
struct track_generation final {
  void operator()(const EvolutionContext& context);
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_LISTENERS_HPP
