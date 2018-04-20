#ifndef VRP_UTILS_POPULATIONFACTORY_HPP
#define VRP_UTILS_POPULATIONFACTORY_HPP

#include "algorithms/Distances.cu"
#include "heuristics/NearestNeighbor.hpp"
#include "solver/genetic/Populations.hpp"
#include "streams/input/SolomonReader.hpp"

#include "test_utils/SolomonBuilder.hpp"

namespace vrp {
namespace test {

template<typename Heuristic = vrp::heuristics::NearestNeighbor>
vrp::models::Tasks createPopulation(std::istream &stream, int populationSize = 3) {
  auto problem = vrp::streams::SolomonReader().read(stream, vrp::algorithms::cartesian_distance());
  return vrp::genetic::create_population<Heuristic>(problem)({populationSize});
}

}
}

#endif //VRP_UTILS_POPULATIONFACTORY_HPP
