#include "algorithms/construction/heuristics/BlinkInsertion.hpp"

#include "streams/in/Solomon.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

using namespace vrp::algorithms::construction;
using namespace vrp::models::costs;
using namespace vrp::models::solution;
using namespace vrp::streams::in;

#include <catch/catch.hpp>

namespace vrp::test {

SCENARIO("blink insertion handles solomon set problems", "[algorithms][construction][insertion]") {
  GIVEN("c101_25 problem") {
    auto stream = create_c101_25_problem_stream{}();
    auto problem = read_solomon_type<cartesian_distance>{}.operator()(stream);
    auto ctx = vrp::test::test_build_insertion_context{}
                 .jobs(problem.jobs->all())
                 .registry(std::make_shared<Registry>(*problem.fleet))
                 .constraint(problem.constraint)
                 .owned();
    auto evaluator = InsertionEvaluator{};

    WHEN("calculates solution") {
      auto solution = BlinkInsertion<>{evaluator}.operator()(ctx);

      THEN("has expected solution") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(!solution.routes.empty());
        REQUIRE(!solution.routes.empty());
      }
    }
  }
}
}