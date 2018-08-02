#ifndef VRP_UTILS_POPULATIONFACTORY_HPP
#define VRP_UTILS_POPULATIONFACTORY_HPP

#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "models/Solution.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/SettingsFactory.hpp"
#include "test_utils/SolomonBuilder.hpp"

namespace vrp {
namespace test {

/// Creates population from input stream. Result includes problem definition.
template<typename Heuristic = vrp::algorithms::heuristics::nearest_neighbor<
           vrp::algorithms::heuristics::TransitionOperator>>
vrp::models::Solution createPopulation(std::istream& stream, int populationSize = 3) {
  auto problem =
    vrp::streams::SolomonReader().read(stream, vrp::algorithms::distances::cartesian_distance());
  auto tasks = vrp::algorithms::genetic::create_population<Heuristic>(problem)(
    createGeneticSettings(populationSize));
  return vrp::models::Solution(std::move(problem), std::move(tasks));
}

}  // namespace test
}  // namespace vrp

#endif  // VRP_UTILS_POPULATIONFACTORY_HPP
