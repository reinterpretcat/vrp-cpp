#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/genetic/Models.hpp"
#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/genetic/Selection.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/ProblemStreams.hpp"

#include <catch/catch.hpp>
#include <vector>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::genetic;
using namespace vrp::streams;
using namespace vrp::runtime;
using namespace vrp::test;

SCENARIO("Can use selection with empty operators", "[genetic][selection]") {
  auto stream = create_c101_problem_stream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  auto selection = Selection{2, {2, {}}, {2, {}}};
  auto ctx = EvolutionContext{0, {}, vector<thrust::pair<int, float>>(12)};

  select_individuums<empty_crossover, empty_mutator>()(ctx, selection);
}
