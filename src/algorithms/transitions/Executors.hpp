#ifndef VRP_ALGORITHMS_TRANSITIONS_EXECUTORS_HPP
#define VRP_ALGORITHMS_TRANSITIONS_EXECUTORS_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {
namespace transitions {

/// Performs transition with a cost.
struct perform_transition final {
  const vrp::models::Problem::Shadow problem;
  const vrp::models::Tasks::Shadow tasks;

  /// Performs transition and returns next task index from which transition should start.
  ANY_EXEC_UNIT int operator()(const vrp::models::Transition& transition, float cost) const;

  /// Performs transition by updating existing state. Returns next task index from which
  /// transition should start.
  ANY_EXEC_UNIT int operator()(const vrp::models::Transition& transition,
                               vrp::models::Transition::State& state) const;
};

}  // namespace transitions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_TRANSITIONS_EXECUTORS_HPP
