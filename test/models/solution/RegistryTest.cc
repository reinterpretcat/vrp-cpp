#include "models/solution/Registry.hpp"

#include "streams/in/Solomon.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::streams::in;
using namespace ranges;

namespace vrp::test {

SCENARIO("registry can provide available actors", "[models][solution][registry]") {
  GIVEN("registry with one driver and multiple vehicles") {
    auto fleet = Fleet{};
    fleet.add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{0, {}, {0, 100}}, {1, {}, {0, 50}}}).owned());

    auto registry = Registry(fleet);

    WHEN("all available actors requested") {
      THEN("then returns three actors") { REQUIRE(ranges::distance(registry.available()) == 3); }
    }

    WHEN("one is used and all available actors requested") {
      ranges::for_each(registry.available() | view::take(1), [&](const auto& actor) { registry.use(actor); });

      THEN("then returns two actors") { REQUIRE(ranges::distance(registry.available()) == 2); }
    }

    WHEN("all are used and all available actors requested") {
      ranges::for_each(registry.available(), [&](const auto& actor) { registry.use(actor); });

      THEN("then returns zero actors") { REQUIRE(ranges::distance(registry.available()) == 0); }
    }
  }
}

SCENARIO("registry can provide unique actors", "[models][solution][registry]") {
  GIVEN("registry with one driver and multiple vehicles") {
    auto fleet = Fleet{};
    fleet.add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{0, {}, {0, 100}}, {1, {}, {0, 50}}}).owned())
      .add(test_build_vehicle{}.id("v3").details({{0, {}, {0, 100}}}).owned());

    auto registry = Registry(fleet);

    WHEN("next actors requested") {
      auto actors = registry.next() | to_vector;
      auto ids = actors | view::transform([](const auto& a) { return a->vehicle->id; }) | to_vector;

      THEN("then returns two unique actors") {
        REQUIRE(actors.size() == 2);
        REQUIRE(actors.front()->detail.start == 0);
        REQUIRE(actors.back()->detail.start == 1);
        CHECK_THAT(ids, Catch::Matchers::Equals(std::vector<std::string>{"v3", "v2"}));
      }
    }
  }

  GIVEN("c101 problem with 25 customers") {
    auto stream = create_c101_25_problem_stream{}();
    auto problem = read_solomon_type<cartesian_distance>{}.operator()(stream);
    auto registry = Registry(*problem.fleet);

    WHEN("unique actors requested") {
      THEN("then returns one actor") { REQUIRE(ranges::distance(registry.next()) == 1); }
    }
  }
}
}
