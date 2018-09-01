#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Models.hpp"
#include "algorithms/genetic/Strategies.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"

#include <catch/catch.hpp>
#include <thrust/iterator/counting_iterator.h>

using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::heuristics;
using namespace vrp::runtime;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::test;

SCENARIO("Can use linear strategy's selection and next", "[genetic][strategy]") {
  auto stream = create_sequential_problem_stream{}();
  auto solution = createPopulation<nearest_neighbor<TransitionOperator>>(stream, 2);
  auto strategy = GuidedStrategy{{16}};
  auto ctx = EvolutionContext{0, std::move(solution),
                              vector<thrust::pair<int, float>>(static_cast<size_t>(16)),
                              thrust::minstd_rand()};

  std::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(5), [&](int i) {
    strategy.selection(ctx);
    strategy.next(ctx);
  });
}
