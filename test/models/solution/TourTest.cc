#include "models/solution/Tour.hpp"

#include "models/extensions/solution/Helpers.hpp"
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

    tour.start(
      build_activity{}.detail({DefaultJobLocation, DefaultDuration, DefaultTimeWindow}).schedule({0, 0}).shared());

    WHEN("activity with service job is added") {
      tour.insert(DefaultActivity);

      THEN("jobs has only one job") {
        auto actual = size(tour.jobs());

        REQUIRE(1 == actual);
      }

      THEN("activities has only two activities") {
        auto actual = size(tour.activities());

        REQUIRE(2 == actual);
      }

      THEN("jobs returns range with original job") {
        std::vector<Job> actual = tour.jobs() | ranges::view::take(1) | ranges::to_vector;

        REQUIRE(DefaultService == actual[0]);
      }

      THEN("remove activity removes both activity and its job") {
        tour.remove(retrieve_job{}(*DefaultActivity).value());

        REQUIRE(0 == size(tour.jobs()));
        REQUIRE(1 == size(tour.activities()));
      }
    }
  }
}
}
