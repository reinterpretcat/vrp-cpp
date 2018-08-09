#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Listeners.hpp"
#include "algorithms/genetic/Models.hpp"
#include "algorithms/genetic/Terminations.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/ProblemStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::genetic;
using namespace vrp::streams;
using namespace vrp::test;

SCENARIO("Can run evolution", "[genetic][evolution]") {
  auto stream = create_c101_problem_stream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  auto settings = Settings{8};

  run_evolution<max_generations, empty_listener>{{5}, {}}(problem, settings);
}