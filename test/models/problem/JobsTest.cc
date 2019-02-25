#include "models/problem/Jobs.hpp"

#include "models/extensions/problem/Factories.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models;
using namespace ranges;

namespace {

struct ProfileAwareTransportCosts final : public TransportCosts {
  vrp::models::common::Duration duration(const std::string& p,
                                         const Location& from,
                                         const Location& to,
                                         const Timestamp&) const override {
    return measure<vrp::models::common::Duration>(p, from, to);
  }

  Distance distance(const std::string& p, const Location& from, const Location& to, const Timestamp&) const override {
    return measure<Distance>(p, from, to);
  }


private:
  template<typename Unit>
  Unit measure(const std::string& profile, const Location& from, const Location& to) const {
    auto value = static_cast<Unit>(to > from ? to - from : from - to);
    return profile == "p2" ? 10 - value : value;
  }
};
}

namespace vrp::test {

SCENARIO("job neighbourhood", "[algorithms][ruin][jobs]") {
  GIVEN("fleet with two profiles and multiple jobs") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").profile("p1").details({{0, 0, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").profile("p2").details({{0, 0, {0, 100}}}).owned());
    auto species = std::vector<models::problem::Job>{as_job(test_build_service{}.location(0).id("s0").shared()),
                                                     as_job(test_build_service{}.location(1).id("s1").shared()),
                                                     as_job(test_build_service{}.location(2).id("s2").shared()),
                                                     as_job(test_build_service{}.location(3).id("s3").shared()),
                                                     as_job(test_build_service{}.location(4).id("s4").shared())};
    auto jobs = Jobs{ProfileAwareTransportCosts{}, *fleet, ranges::view::all(species)};

    auto [index, expected] = GENERATE(table<int, std::vector<std::string>>({
      {0, {"s1", "s2", "s3", "s4"}},
      {1, {"s0", "s2", "s3", "s4"}},
      {2, {"s1", "s3", "s0", "s4"}},
      {3, {"s2", "s4", "s1", "s0"}},
    }));

    WHEN("get neighbours for specific profile") {
      auto result = jobs.neighbors("p1", species.at(index), Timestamp{}) |
        view::transform([](const auto& j) { return get_job_id{}(j); }) | to_vector;

      THEN("returns expected jobs") { CHECK_THAT(result, Catch::Matchers::Equals(expected)); }
    }
  }
}

SCENARIO("job rank", "[algorithms][ruin][jobs]") {
  GIVEN("fleet with two profiles and multiple jobs") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1_1").profile("p1").details({{0, 0, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v1_2").profile("p1").details({{15, 0, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2_1").profile("p3").details({{30, 0, {0, 100}}}).owned());
    auto species = std::vector<models::problem::Job>{as_job(test_build_service{}.location(0).id("s0").shared()),
                                                     as_job(test_build_service{}.location(10).id("s1").shared()),
                                                     as_job(test_build_service{}.location(21).id("s2").shared()),
                                                     as_job(test_build_service{}.location(31).id("s3").shared())};
    auto jobs = Jobs{ProfileAwareTransportCosts{}, *fleet, ranges::view::all(species)};

    auto [index, profile, expected] = GENERATE(table<int, std::string, common::Distance>({
      {0, "p1", 0},
      {1, "p1", 5},
      {2, "p1", 6},
      {3, "p1", 16},
      {0, "p3", 30},
      {1, "p3", 20},
      {2, "p3", 9},
      {3, "p3", 1},
    }));

    WHEN("get rank for specific job and profile") {
      auto result = jobs.rank(profile, species.at(index));

      THEN("returns expected jobs") { REQUIRE(result == expected); }
    }
  }
}
}