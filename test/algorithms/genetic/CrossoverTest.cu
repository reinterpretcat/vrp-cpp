#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "test_utils/MemoryUtils.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"
#include "test_utils/VectorUtils.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::utils;
using namespace vrp::test;

namespace {
Solution getPopulation(int populationSize) {
  auto stream = create_c101_problem_stream{}();
  return createPopulation<>(stream, populationSize);
};

/// Runs crossover
struct run_crossover final {
  Solution::Shadow solution;
  DevicePool::Pointer pool;
  const Settings settings;
  const Generation generation;
  __device__ void operator()(int index) {
    adjusted_cost_difference<nearest_neighbor>{solution, pool}(settings, generation);
  }
};

}  // namespace

SCENARIO("Can create offsprings", "[genetic][crossover][acdc]") {
  int populationSize = 4;
  auto solution = getPopulation(populationSize);
  auto settings = Settings{populationSize, {0.75, 0.05}};
  auto generation = Generation{{0, 1}, {2, 3}};
  auto pool = vrp::utils::DevicePool::create(1, 4, 1000);

  thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1),
                   run_crossover{solution.getShadow(), *pool, settings, generation});

  REQUIRE(SolutionChecker::check(solution).isValid());
  CHECK_THAT(
    vrp::test::copy(solution.tasks.ids),
    Catch::Matchers::Equals(std::vector<int>{0, 1,  20, 21, 22, 23, 2,  24, 25, 10, 11, 9,  6,
                                             4, 5,  3,  7,  8,  12, 13, 17, 18, 19, 15, 16, 14,

                                             0, 7,  4,  1,  20, 21, 22, 23, 2,  24, 25, 10, 11,
                                             9, 6,  5,  3,  8,  12, 13, 17, 18, 19, 15, 16, 14,

                                             0, 20, 21, 22, 23, 2,  24, 25, 10, 11, 9,  6,  4,
                                             5, 3,  8,  1,  7,  12, 13, 17, 18, 19, 15, 16, 14,

                                             0, 20, 21, 22, 23, 2,  24, 25, 10, 11, 9,  6,  4,
                                             5, 3,  8,  1,  7,  12, 13, 17, 18, 19, 15, 16, 14}));
}
