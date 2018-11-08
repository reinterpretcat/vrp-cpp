#include "models/problem/Service.hpp"
#include "models/solution/Tour.hpp"

#include "testUtils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace vrp::models::solution;

namespace vrp::test {

SCENARIO("tour can handle activities with job relations", "[models][tour]") {
  GIVEN("A tour model") {
    auto tour = Tour();

    WHEN("activity with service job is added") {
      tour.add(DefaultActivity);

      THEN("jobs returns sequence with original job") {

      }

      THEN("activities returns sequence with original activity") {

      }
    }
  }
}

}
