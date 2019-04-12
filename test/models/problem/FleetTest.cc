#include "models/problem/Fleet.hpp"

#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace Catch::Matchers;

namespace vrp::test {

SCENARIO("fleet model", "[models][problem][fleet]") {
  GIVEN("fleet with different vehicles") {
    auto fleet = Fleet{};
    fleet.add(test_build_vehicle{}.id("v1").profile(DefaultProfile + 1).details({{0, 0, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").profile(DefaultProfile + 2).details({{0, 0, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v3").profile(DefaultProfile + 1).details({{0, 0, {0, 100}}}).owned());

    WHEN("get profiles") {
      auto profiles = fleet.profiles() | ranges::to_vector;

      THEN("return unique profiles") { CHECK_THAT(profiles, Equals(std::vector<models::common::Profile>{1, 2})); }
    }
  }
}
}
