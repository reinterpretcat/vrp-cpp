#include "Solver.hpp"

#include "streams/in/Solomon.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms;
using namespace vrp::algorithms::construction;
using namespace vrp::streams::in;

namespace vrp::test {

SCENARIO("DefaultSolver can solve c101 problem", "[solver][default]") {
  auto solver = DefaultSolver{};

  GIVEN("C101 problem with 25 customers") {
    auto stream = create_c101_25_problem_stream{}();
    auto problem = read_solomon_type<cartesian_distance<1>>{}.operator()(stream);

    WHEN("run solver") {
      auto estimatedSolution = solver(problem);

      THEN("has valid solution") {
        REQUIRE(estimatedSolution.first->unassigned.empty());
        REQUIRE(estimatedSolution.first->routes.size() <= 6);
      }
    }
  }
}
}
