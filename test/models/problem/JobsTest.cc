#include "models/problem/Jobs.hpp"

#include "models/extensions/problem/Factories.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models;

namespace {

struct ProfileAwareTransportCosts final : public TransportCosts {
  Duration duration(const std::string& p, const Location& from, const Location& to, const Timestamp&) const override {
    return measure<Duration>(p, from, to);
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
    auto profiles = fleet->profiles();
    auto species = std::vector<models::problem::Job>{as_job(test_build_service{}.location(0).id("s0").shared()),
                                                     as_job(test_build_service{}.location(1).id("s1").shared()),
                                                     as_job(test_build_service{}.location(2).id("s2").shared()),
                                                     as_job(test_build_service{}.location(3).id("s3").shared()),
                                                     as_job(test_build_service{}.location(4).id("s4").shared())};
    auto jobs = Jobs{ProfileAwareTransportCosts{}, ranges::view::all(species), profiles};

    WHEN("get neighbours for specific profile") {
      auto result = jobs.neighbors("p1", species.at(0), Timestamp{}, [](const auto d) { return d > 0; });

      THEN("returns expected jobs") {}
    }
  }
}
}