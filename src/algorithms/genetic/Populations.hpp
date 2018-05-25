#ifndef VRP_ALGORITHMS_GENETIC_POPULATIONS_HPP
#define VRP_ALGORITHMS_GENETIC_POPULATIONS_HPP

#include "algorithms/genetic/Models.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"

namespace vrp {
namespace algorithms {
namespace genetic {

/// Creates initial population based on problem definition
/// and settings using fast heuristic provided.
template<typename Heuristic>
struct create_population final {
  const vrp::models::Problem& problem;

  explicit create_population(const vrp::models::Problem& problem) : problem(problem) {}

  vrp::models::Tasks operator()(const Settings& settings);
};

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_GENETIC_POPULATIONS_HPP
