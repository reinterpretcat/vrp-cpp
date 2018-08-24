#ifndef VRP_UTILS_POPULATIONFACTORY_HPP
#define VRP_UTILS_POPULATIONFACTORY_HPP

#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "models/Solution.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/SolomonBuilder.hpp"

namespace vrp {
namespace test {

/// Creates problem.
template<typename Problem>
vrp::models::Problem createProblem() {
  auto stream = Problem{}();
  return vrp::streams::SolomonReader().read(stream,
                                            vrp::algorithms::distances::cartesian_distance());
}

/// Creates population from input stream. Result includes problem definition.
template<typename Heuristic = vrp::algorithms::heuristics::nearest_neighbor<
           vrp::algorithms::heuristics::TransitionOperator>>
vrp::models::Solution createPopulation(std::istream& stream, int populationSize = 3) {
  auto problem =
    vrp::streams::SolomonReader().read(stream, vrp::algorithms::distances::cartesian_distance());
  auto tasks = vrp::algorithms::genetic::create_population<Heuristic>{problem}(populationSize);
  return vrp::models::Solution(std::move(problem), std::move(tasks));
}

/// Creates population from problem and task's data.
inline vrp::models::Solution createPopulation(
  vrp::models::Problem&& problem,
  const std::initializer_list<int>& ids,
  const std::initializer_list<float>& costs,
  const std::initializer_list<int>& times,
  const std::initializer_list<int>& capacities,
  const std::initializer_list<int>& vehicles,
  const std::initializer_list<vrp::models::Plan>& plan) {
  auto tasks = vrp::models::Tasks();
  tasks.customers = problem.size();
  tasks.ids = std::vector<int>{ids};
  tasks.costs = std::vector<float>{costs};
  tasks.times = std::vector<int>{times};
  tasks.capacities = std::vector<int>{capacities};
  tasks.vehicles = std::vector<int>{vehicles};
  tasks.plan = std::vector<vrp::models::Plan>{plan};
  return vrp::models::Solution(std::move(problem), std::move(tasks));
}

}  // namespace test
}  // namespace vrp

#endif  // VRP_UTILS_POPULATIONFACTORY_HPP
