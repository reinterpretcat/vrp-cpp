#ifndef VRP_SOLVER_GENETIC_POPULATIONS_HPP
#define VRP_SOLVER_GENETIC_POPULATIONS_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "solver/genetic/Settings.hpp"

namespace vrp {
namespace genetic {

/// Creates initial population based on problem definition
/// and settings using fast heuristic provided.
template <typename Heuristic>
struct create_population final {
  const vrp::models::Problem &problem;

  explicit create_population(const vrp::models::Problem &problem) :
      problem(problem) {}

  vrp::models::Tasks operator()(const vrp::genetic::Settings &settings);
};

}
}

#endif //VRP_SOLVER_GENETIC_POPULATIONS_HPP
