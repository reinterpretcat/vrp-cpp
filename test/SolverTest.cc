#include "streams/in/scientific/LiLim.hpp"
#include "streams/in/scientific/Solomon.hpp"
#include "test_utils/Solvers.hpp"
#include "test_utils/algorithms/refinement/LogAndValidate.hpp"
#include "test_utils/streams/LiLimStreams.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms;
using namespace vrp::algorithms::construction;
using namespace vrp::streams::in;

namespace {
auto solver = vrp::test::create_default_solver<vrp::test::log_and_validate>{}();
}

namespace vrp::test {

SCENARIO("Solver can solve C101 problem greedy acceptance and default RaR", "[solver][default]") {
  GIVEN("C101 problem with 25 customers") {
    auto stream = create_c101_25_problem_stream{}();
    auto problem = read_solomon_type<cartesian_distance>{}.operator()(stream);

    WHEN("run solver") {
      auto estimatedSolution = solver(problem);

      THEN("has valid solution") {
        REQUIRE(estimatedSolution.first->unassigned.empty());
        REQUIRE(estimatedSolution.first->routes.size() == 3);
      }
    }
  }
}

SCENARIO("Solver can solve LC101 problem greedy acceptance and default RaR", "[solver][default]") {
  GIVEN("LC101 problem with 53 sequences") {
    auto stream = create_lc101_problem_stream{}();
    auto problem = read_li_lim_type<cartesian_distance>{}.operator()(stream);

    WHEN("run solver") {
      auto estimatedSolution = solver(problem);

      THEN("has valid solution") {
        REQUIRE(estimatedSolution.first->unassigned.empty());
        REQUIRE(estimatedSolution.first->routes.size() == 10);
        REQUIRE(std::abs(estimatedSolution.second.total() - 828.937) < 0.001);
      }
    }
  }
}
}
