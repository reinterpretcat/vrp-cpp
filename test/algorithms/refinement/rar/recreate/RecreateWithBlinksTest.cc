#include "algorithms/refinement/rar/recreate/RecreateWithBlinks.hpp"

#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "streams/in/Solomon.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Extensions.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

using namespace vrp::algorithms::construction;
using namespace vrp::algorithms::refinement;
using namespace vrp::algorithms::objectives;
using namespace vrp::models;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::streams::in;
using namespace ranges;

#include <catch/catch.hpp>

namespace vrp::test {

SCENARIO("recreate with blinks handles simple problem", "[algorithms][refinement][recreate]") {
  GIVEN("one service job and two vehicles") {
    auto fleet = Fleet{};
    fleet.add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{20, {}, {0, 100}}}).owned());

    auto recreate = RecreateWithBlinks{};

    WHEN("analyzes insertion context") {
      auto problem = std::make_shared<Problem>(Problem{{},
                                                       {},
                                                       {},
                                                       std::make_shared<penalize_unassigned_jobs<>>(),
                                                       std::make_shared<ActivityCosts>(),
                                                       std::make_shared<TestTransportCosts>()});
      auto result = recreate({problem, {}, {}},
                             test_build_insertion_context{}
                               .registry(std::make_shared<Registry>(fleet))
                               .constraint(std::make_shared<InsertionConstraint>())
                               .jobs({as_job(test_build_service{}.location(3).shared())})
                               .owned());

      THEN("returns new solution with job inserted") {
        REQUIRE(result.unassigned.empty());
        REQUIRE(result.routes.size() == 1);
        REQUIRE(result.routes.front()->actor->vehicle->id == "v1");
        REQUIRE(result.routes.front()->tour.get(0)->detail.location == 3);
      }
    }
  }
}
}