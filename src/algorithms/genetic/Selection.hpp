#ifndef VRP_ALGORITHMS_GENETIC_SELECTION_HPP
#define VRP_ALGORITHMS_GENETIC_SELECTION_HPP

#include "algorithms/genetic/Models.hpp"
#include "models/Solution.hpp"

#include <thrust/pair.h>
#include <thrust/random/linear_congruential_engine.h>

namespace vrp {
namespace algorithms {
namespace genetic {

/// Selects individuums for evolution.
template<typename Crossover, typename Mutator>
struct select_individuums final {
  Crossover crossover;
  Mutator mutator;
  thrust::minstd_rand rng;
  void operator()(const Selection& selection);
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_SELECTION_HPP
