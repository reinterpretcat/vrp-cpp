#include "models/problem/Service.hpp"
#include "models/solution/Tour.hpp"

#include "test_utils/models/Factories.hpp"
#include "test_utils/models/Matchers.hpp"

#include <catch/catch.hpp>

using namespace ranges::v3;
using namespace vrp::models::problem;
using namespace vrp::models::solution;

namespace vrp::test {

SCENARIO("tour can handle activities with job relations", "[models][tour]") {

  GIVEN("A tour model") {
    auto tour = Tour();

    WHEN("activity with service job is added") {
      tour.add(DefaultActivity);

      THEN("has only one job") {
        auto actual = size(tour.jobs());

        REQUIRE(1 == actual);
      }

      THEN("has only one activity") {
        auto actual = size(tour.activities());

        REQUIRE(1 == actual);
      }

      THEN("returns sequence with original job") {
        std::vector<Activity::Job> actual = tour.jobs() | view::take(1);

        REQUIRE(DefaultService == actual[0]);
      }

      THEN("returns sequence with original activity") {
        std::vector<Activity> actual = tour.activities() | view::take(1);

        CHECK_THAT(actual[0], ActivityMatcher(DefaultActivity));
      }
    }
  }
}

}
