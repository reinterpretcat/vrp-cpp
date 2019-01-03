#include "algorithms/refinement/rr/recreate/RecreateWithBlinks.hpp"

#include "streams/in/Solomon.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Extensions.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

using namespace vrp::algorithms::construction;
using namespace vrp::models::costs;
using namespace vrp::models::solution;
using namespace vrp::streams::in;

#include <catch/catch.hpp>

namespace vrp::test {

SCENARIO("recreate with blinks handles simple problem", "[algorithms][refinement][recreate]") {
  //  GIVEN("c101_25 problem") {
  //    auto stream = create_c101_25_problem_stream{}();
  //    auto problem = read_solomon_type<cartesian_distance<1>>{}.operator()(stream);
  //
  //    WHEN("calculates solution") {
  //      auto solution = BlinkInsertion<>{evaluator}.operator()(ctx);
  //
  //      THEN("has expected solution") {
  //        REQUIRE(solution.jobs.empty());
  //        REQUIRE(solution.unassigned.empty());
  //        REQUIRE(!solution.routes.empty());
  //        REQUIRE(!solution.routes.empty());
  //      }
  //    }
  //  }
}
}