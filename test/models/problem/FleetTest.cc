#include "models/problem/Fleet.hpp"

#include <catch/catch.hpp>
#include "test_utils/models/Factories.hpp"

using namespace vrp::models::problem;
using namespace Catch::Matchers;

namespace vrp::test {

SCENARIO("fleet model", "[models][problem][fleet]") {
  GIVEN("fleet with different vehicles") {
    auto fleet = Fleet{}
        .add(test_build_vehicle{}.id("v1").profile("1").details({{0, 0, {0, 100}}}).owned())
        .add(test_build_vehicle{}.id("v2").profile("2").details({{0, 0, {0, 100}}}).owned())
        .add(test_build_vehicle{}.id("v3").profile("1").details({{0, 0, {0, 100}}}).owned());

    WHEN("get profiles") {
      auto profiles = fleet.profiles() | ranges::to_vector;

      THEN("return unique profiles") {
        CHECK_THAT(profiles, Equals(std::vector<std::string>{"1", "2"}));
      }
    }
  }
}

}
