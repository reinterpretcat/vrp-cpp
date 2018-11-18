#include "algorithms/construction/insertion/constraints/VehicleActivityTiming.hpp"

#include "models/costs/ActivityCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "test_utils/algorithms/construction/Contexts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>
#include <utility>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;
using namespace vrp::models::problem;

namespace vrp::test {

SCENARIO("vehicle activity timing accept method", "[algorithms][construction][insertion]") {
  GIVEN("tour with two activities") {
    auto prev = test_build_activity{}.location(10).duration(0).schedule({10, 10}).shared();
    auto next = test_build_activity{}.location(20).duration(0).schedule({20, 20}).shared();
    auto route = test_build_route{}.owned();
    route.tour.add(prev).add(next);

    WHEN("accept with simple fleet") {
      auto fleet = std::make_shared<Fleet>();
      fleet->add(*DefaultActor->driver).add(*DefaultActor->vehicle);
      auto state = InsertionRouteState{};

      VehicleActivityTiming(fleet,
                            std::make_shared<TestTransportCosts>(),  //
                            std::make_shared<ActivityCosts>())
        .accept(route, state);

      THEN("route state state is properly updated") {}
    }
  }
}
}
