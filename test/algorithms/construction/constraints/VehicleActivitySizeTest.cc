#include "algorithms/construction/constraints/VehicleActivitySize.hpp"

#include "algorithms/construction/extensions/States.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "test_utils/algorithms/constraints/Helpers.hpp"
#include "test_utils/algorithms/construction/Contexts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;
using namespace vrp::models::problem;
using namespace vrp::test;

namespace {
Tour::Activity
activity(int size) {
  return test_build_activity{}.job(as_job(test_build_service{}.dimens({{"size", size}}).shared())).shared();
}
}

namespace vrp::test {

SCENARIO("vehicle activity size", "[algorithms][construction][constraints]") {
  GIVEN("fleet with 1 vehicle and service jobs") {
    auto fleet = std::make_shared<Fleet>();
    fleet->add(test_build_vehicle{}.id("v1").dimens({{"size", 10}}).details(asDetails(0, {}, {0, 100})).owned());

    WHEN("accept route with service") {
      auto state = InsertionRouteState{};
      auto route = test_build_route{}.actor(getActor("v1", *fleet)).shared();
      route->tour.add(activity(-1)).add(activity(2)).add(activity(-3));

      //VehicleActivitySize<int>{}.accept(*route, state);

      // TODO
    }
  }
}
}
