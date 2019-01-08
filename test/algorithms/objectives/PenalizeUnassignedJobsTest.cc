#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"

#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::algorithms::objectives;
using namespace vrp::models;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::utils;

namespace {

inline auto
createRoute(const vrp::models::solution::Route::Actor& actor, const vrp::models::common::Schedule& endSchedule) {
  using namespace vrp::test;

  return test_build_route{}
    .actor(actor)
    .start(test_build_activity{}.schedule({0, 0}).type(Activity::Type::Start).shared())
    .end(test_build_activity{}.schedule(endSchedule).type(Activity::Type::End).shared())
    .shared();
}
}

namespace vrp::test {

SCENARIO("penalize unassigned jobs calculates cost properly", "[algorithms][objectives]") {
  auto fleet = Fleet{};
  fleet.add(test_build_driver{}.owned())
    .add(test_build_vehicle{}.id("v1").owned())
    .add(test_build_vehicle{}.id("v2").owned());
  auto registry = std::make_shared<Registry>(fleet);

  GIVEN("solution with two routes and one unassigned job") {
    auto actor1 = getActor("v1", fleet);
    auto actor2 = getActor("v2", fleet);
    registry->use(*actor1);
    registry->use(*actor2);

    auto route1 = createRoute(actor1, {40, 40});
    route1
      ->tour  //
      .add(test_build_activity{}.detail({10, 5, DefaultTimeWindow}).shared())
      .add(test_build_activity{}.detail({15, 5, DefaultTimeWindow}).shared());

    auto route2 = createRoute(actor1, {11, 11});
    route2->tour.add(test_build_activity{}.detail({5, 1, DefaultTimeWindow}).shared());

    auto problem = Problem{{}, {}, {}, std::make_shared<ActivityCosts>(), std::make_shared<TestTransportCosts>()};
    auto solution = Solution{registry, {route1, route2}, {{DefaultService, 0}}};

    WHEN("calculates objective costs") {
      auto [actual, penalty] = penalize_unassigned_jobs<>{}(problem, solution);

      THEN("actual cost is correct") { REQUIRE(actual == 251); }

      THEN("penalty cost is correct") { REQUIRE(penalty == 1000); }
    }
  }
}
}