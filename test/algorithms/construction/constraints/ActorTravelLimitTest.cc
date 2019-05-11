#include "algorithms/construction/constraints/ActorTravelLimit.hpp"

#include "models/costs/ActivityCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"
#include "test_utils/algorithms/construction/Factories.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/models/Helpers.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models;
using namespace vrp::models::costs;
using namespace vrp::models::common;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::test;
using namespace ranges;
using namespace Catch::Generators;

using Limit = ActorTravelLimit::Limit;
using DistLimit = Limit::Distance;
using DurLimit = Limit::Duration;

namespace {

struct TestDistanceTransportCosts : public TestTransportCosts {
  Duration duration(const Profile profile, const Location& from, const Location& to, const Timestamp&) const override {
    return 1;
  }
};

struct TestDurationTransportCosts : public TestTransportCosts {
  Duration distance(const Profile profile, const Location& from, const Location& to, const Timestamp&) const override {
    return 1;
  }
};

constexpr static int DistanceCode = 1;
constexpr static int DurationCode = 2;
}

namespace vrp::test {

SCENARIO("actor travel limit can manage distance limit", "[algorithms][construction][constraints]") {
  auto [transport, target, used, value, dist, dur, expected] = GENERATE(table<std::shared_ptr<TestTransportCosts>,
                                                                              std::string,
                                                                              std::string,
                                                                              double,
                                                                              DistLimit,
                                                                              DurLimit,
                                                                              HardActivityConstraint::Result>({
    {std::make_shared<TestDistanceTransportCosts>(), "v1", "v1", 76, {100}, {}, stop(DistanceCode)},
    {std::make_shared<TestDistanceTransportCosts>(), "v1", "v1", 74, {100}, {}, success()},
    {std::make_shared<TestDistanceTransportCosts>(), "v1", "v2", 76, {100}, {}, success()},

    {std::make_shared<TestDurationTransportCosts>(), "v1", "v1", 76, {}, {100}, stop(DurationCode)},
    {std::make_shared<TestDurationTransportCosts>(), "v1", "v1", 74, {}, {100}, success()},
    {std::make_shared<TestDurationTransportCosts>(), "v1", "v2", 76, {}, {100}, success()},
  }));

  GIVEN("fleet with one vehicle") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").owned())
      .add(test_build_vehicle{}.id("v2").owned());
    auto registry = std::make_shared<Registry>(*fleet);

    WHEN("has distance limit for the actor") {
      auto routeState = std::make_shared<InsertionRouteState>();
      routeState->put<Distance>(ActorTravelLimit::DistanceKey, 50);
      routeState->put<Duration>(ActorTravelLimit::DurationKey, 50);

      auto slnCtx = InsertionSolutionContext{{}, {}, {}, {}, registry};
      auto actCtx = test_build_insertion_activity_context{}
                      .prev(test_build_activity{}.location(0).shared())
                      .target(test_build_activity{}.location(value).shared())
                      .next(test_build_activity{}.location(50).shared())
                      .owned();

      auto actorTravelLimit = ActorTravelLimit{
        {Limit{[target = target](const auto& a) { return get_vehicle_id{}(*a.vehicle) == target; }, dist, dur}},
        transport,
        std::make_shared<ActivityCosts>(),
        DistanceCode,
        DurationCode};

      actorTravelLimit.accept(slnCtx);

      THEN("returns expected constraint check") {
        auto result = actorTravelLimit.hard(
          InsertionRouteContext{std::make_shared<Route>(Route{get_actor_from_registry{}(used, *registry), {}}),
                                routeState},
          actCtx);

        REQUIRE(result == expected);
      }
    }
  }
}
}