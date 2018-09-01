#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Models.hpp"
#include "algorithms/genetic/Strategies.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/ProblemStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::genetic;
using namespace vrp::streams;
using namespace vrp::test;

SCENARIO("Can run evolution", "[genetic][evolution]") {
  auto stream = rc1_10_1_problem_stream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());

  run_evolution<GuidedStrategy>{{16}}(problem);
}
