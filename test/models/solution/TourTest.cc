#include "models/solution/Tour.hpp"

#include "models/problem/Service.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/models/Matchers.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace vrp::models::solution;

namespace vrp::test {

SCENARIO("tour can handle activities with job relations", "[models][tour]") {
  GIVEN("A tour model") {
    auto tour = Tour();

    WHEN("activity with service job is added") {
      tour.add(DefaultActivity);

      THEN("jobs has only one job") {
        auto actual = size(tour.jobs());

        REQUIRE(1 == actual);
      }

      THEN("activities has only one activity") {
        auto actual = size(tour.activities());

        REQUIRE(1 == actual);
      }

      THEN("jobs returns range with original job") {
        std::vector<Job> actual = tour.jobs() | ranges::view::take(1);

        REQUIRE(DefaultService == actual[0]);
      }

      THEN("activities returns range with original activity") {
        std::vector<Tour::Activity> actual = tour.activities() | ranges::view::take(1);

        CHECK_THAT(*actual[0], ActivityMatcher(*DefaultActivity));
      }

      THEN("remove activity removes both activity and its job") {
        tour.remove(DefaultActivity->job.value());

        REQUIRE(0 == size(tour.jobs()));
        REQUIRE(0 == size(tour.activities()));
      }
    }
  }
}
}
