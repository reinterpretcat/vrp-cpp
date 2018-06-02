#ifndef VRP_MODELS_SOLUTION_HPP
#define VRP_MODELS_SOLUTION_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"

namespace vrp {
namespace models {

/// Represents VRP solution.
struct Solution final {
  /// Stores shadow data.
  struct Shadow final {
    Problem::Shadow problem;
    Tasks::Shadow tasks;
  };

  Solution(Problem&& problem, Tasks&& tasks) :
    problem(std::move(problem)), tasks(std::move(tasks)){};

  // Disable copying
  Solution(const Solution&) = delete;
  Solution& operator=(const Solution&) = delete;

  // Allow move
  Solution(Solution&&) = default;

  /// Problem definition.
  const Problem problem;

  /// Assigned tasks.
  Tasks tasks;

  /// Returns shadow object.
  Shadow getShadow() { return {problem.getShadow(), tasks.getShadow()}; }
};

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_SOLUTION_HPP
