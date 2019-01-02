#include "models/solution/Registry.hpp"

#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace ranges;

namespace vrp::test {

SCENARIO("registry can provide actors", "[models][solution][registry]") {
  GIVEN("registry with one driver and multiple vehicles") {
    auto fleet = Fleet{};
    fleet.add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{0, {}, {0, 100}}, {1, {}, {0, 50}}}).owned());

    auto registry = Registry(fleet);

    WHEN("all available actors requested") {
      THEN("then returns three actors") { REQUIRE(ranges::distance(registry.actors()) == 3); }
    }

    WHEN("one is used and all available actors requested") {
      auto actors =
        ranges::for_each(registry.actors() | view::take(1), [&](const auto& actor) { registry.use(*actor); });

      THEN("then returns two actors") { REQUIRE(ranges::distance(registry.actors()) == 2); }
    }

    WHEN("all are used and all available actors requested") {
      auto actors = ranges::for_each(registry.actors(), [&](const auto& actor) { registry.use(*actor); });

      THEN("then returns zero actors") { REQUIRE(ranges::distance(registry.actors()) == 0); }
    }
  }
}
}
