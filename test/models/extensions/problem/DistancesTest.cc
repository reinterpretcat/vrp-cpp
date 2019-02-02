#include "models/extensions/problem/Distances.hpp"

#include "models/extensions/problem/Factories.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace vrp::models::common;
using namespace Catch::Generators;

using JobDetails = std::vector<Service::Detail>;

namespace {

Service::Detail
jobDetail(const std::optional<Location>& location) {
  return {location, 0, {vrp::test::DefaultTimeWindow}};
}
}

namespace vrp::test {

SCENARIO("distance between service jobs", "[models][extensions][problem]") {
  auto [d1, d2, expected] = GENERATE(table<JobDetails, JobDetails, Distance>(
    {{{jobDetail({0})}, {jobDetail({10})}, 10},
     {{jobDetail({0})}, {jobDetail({})}, 0},
     {{jobDetail({})}, {jobDetail({})}, 0},
     {{jobDetail({3})}, {jobDetail({5}), jobDetail({2})}, 1},
     {{jobDetail({2}), jobDetail({1})}, {jobDetail({10}), jobDetail({9})}, 7}}));

  GIVEN("service jobs") {
    auto s1 = as_job(test_build_service{}.details(std::move(d1)).shared());
    auto s2 = as_job(test_build_service{}.details(std::move(d2)).shared());

    WHEN("distance calculated") {
      auto result = job_distance{TestTransportCosts{}, "", Timestamp{}}.operator()(s1, s2);

      THEN("has expected value") { REQUIRE(result == expected); }
    }
  }
}
}