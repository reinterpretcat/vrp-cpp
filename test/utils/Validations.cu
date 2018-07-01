#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include "test_utils/PopulationFactory.hpp"
#include "test_utils/ProblemStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::utils;
using namespace vrp::test;

SCENARIO("Can check valid solution", "[utils][validation][solution_checker]") {
  auto stream = create_sequential_problem_stream()();
  auto population = createPopulation<nearest_neighbor>(stream, 2);

  auto result = SolutionChecker::check(population);

  REQUIRE(result.isValid());
}

SCENARIO("Is","[utils][validation][solution_checker")
{
  //REQUIRE(SolutionChecker::check(solution).isValid());
}