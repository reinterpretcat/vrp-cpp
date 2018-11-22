#include "models/solution/Dispatcher.hpp"

#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace vrp::models::solution;

namespace vrp::test {

SCENARIO("dispatcher can provide actors", "[models][solution][dispatcher]") {
  GIVEN("dispatcher with one driver and multiple vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{0, {}, {0, 100}}, {1, {}, {0, 50}}}).owned());

    auto dispatcher = Dispatcher(fleet);

    WHEN("all available actors requested") {
      THEN("then returns three actors") { REQUIRE(ranges::distance(dispatcher.actors()) == 3); }
    }
  }
}
}
