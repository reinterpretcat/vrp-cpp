#include "models/extensions/problem/Distances.hpp"

#include "models/extensions/problem/Factories.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace vrp::models::common;
using namespace Catch::Generators;

using JobDetails = std::vector<JobDetail>;

namespace {

JobDetail
detail(const std::optional<Location>& location) {
  return {location, 0, {vrp::test::DefaultTimeWindow}};
}
}

namespace vrp::test {

SCENARIO("distance between jobs", "[models][extensions][problem]") {
  GIVEN("service jobs") {
    auto [d1, d2, expected] =
      GENERATE(table<JobDetails, JobDetails, Distance>({{{detail({0})}, {detail({10})}, 10},
                                                        {{detail({0})}, {detail({})}, 0},
                                                        {{detail({3})}, {detail({5}), detail({2})}, 1},
                                                        {{detail({2}), detail({1})}, {detail({10}), detail({9})}, 7}}));

    auto s1 = as_job(test_build_service{}.details(std::move(d1)).shared());
    auto s2 = as_job(test_build_service{}.details(std::move(d2)).shared());

    WHEN("distance calculated") {
      auto result = job_distance{TestTransportCosts{}, "", Timestamp{}}.operator()(s1, s2);

      THEN("has expected value") { REQUIRE(result == expected); }
    }
  }
}
}