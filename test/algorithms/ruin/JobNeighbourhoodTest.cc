#include "algorithms/ruin/JobNeighbourhood.hpp"

#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::ruin;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;

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
    return std::stoi(profile) * static_cast<Unit>(to > from ? to - from : from - to);
  }
};
}

namespace vrp::test {

SCENARIO("job neighbourhood", "[algorithms][ruin][jobs]") {
  GIVEN("fleet and jobs") {
    auto fleet = Fleet{}
                   .add(test_build_driver{}.owned())
                   .add(test_build_vehicle{}.id("v1").profile("1").details({{0, 0, {0, 100}}}).owned())
                   .add(test_build_vehicle{}.id("v2").profile("2").details({{0, 0, {0, 100}}}).owned());
    auto transport = ProfileAwareTransportCosts{};
    auto neighbourhood = JobNeighbourhood{fleet, transport};
  }
}
}