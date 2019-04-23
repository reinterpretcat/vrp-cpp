#include "algorithms/construction/constraints/ActorJobLock.hpp"

#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"
#include "test_utils/algorithms/construction/Factories.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::test;
using namespace ranges;
using namespace Catch::Generators;

namespace {
const Registry::SharedActor
getActorFromRegistry(const std::string& vehicleId, const Registry& registry) {
  return ranges::front(registry.available() |
                       view::remove_if([&](const auto& a) { return get_vehicle_id{}(*a->vehicle) != vehicleId; }) |
                       to_vector);
}

const auto TerminalActivity =
  build_activity{}.detail({DefaultJobLocation, DefaultDuration, DefaultTimeWindow}).schedule({0, 0}).shared();
const auto OtherActivity = test_build_activity{}.shared();

const auto s1 = test_build_service{}.id("s1").shared();
const auto s2 = test_build_service{}.id("s2").shared();
}

namespace vrp::test {

SCENARIO("actor job lock can manage any actor-job locks on route level", "[algorithms][construction][constraints]") {
  auto [locked, used, expected] = GENERATE(table<std::string, std::string, HardRouteConstraint::Result>({
    {"v1", "v1", {}},
    {"v1", "v2", {3}},
  }));

  GIVEN("fleet with 2 vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details(asDetails(0, {}, {0, 100})).owned())
      .add(test_build_vehicle{}.id("v2").details(asDetails(0, {}, {0, 100})).owned());
    auto registry = std::make_shared<Registry>(*fleet);

    WHEN("has job lock for one actor") {
      auto slnCtx = InsertionSolutionContext{{}, {}, {}, {}, registry};
      auto actorJobLock =
        ActorJobLock{{Lock{[locked = locked](const auto& a) { return get_vehicle_id{}(*a.vehicle) == locked; },
                           {Lock::Detail{Lock::Order::Any, Lock::Position::middle(), {DefaultService}}}}}};
      actorJobLock.accept(slnCtx);

      THEN("returns expected constraint check") {
        auto result = actorJobLock.hard(
          InsertionRouteContext{std::make_shared<Route>(Route{getActorFromRegistry(used, *registry), {}}),
                                std::make_shared<InsertionRouteState>()},
          DefaultService);

        REQUIRE(result == expected);
      }
    }
  }
}

SCENARIO("actor job lock can manage strict actor-job locks on activity level",
         "[algorithms][construction][constraints]") {
  auto [message, position, prev, next, expected] =
    GENERATE_REF(table<std::string, Lock::Position, Tour::Activity, Tour::Activity, HardActivityConstraint::Result>({
      //// Middle ////
      {
        "should reject insertion in between within middle position",
        Lock::Position::middle(),
        test_build_activity{}.service(s1).shared(),
        test_build_activity{}.service(s2).shared(),
        {{false, 3}},
      },

      {
        "should allow insertion before strict sequence within middle position",
        Lock::Position::middle(),
        OtherActivity,
        test_build_activity{}.service(s1).shared(),
        {},
      },
      {
        "should allow insertion before strict sequence after departure within middle position",
        Lock::Position::middle(),
        TerminalActivity,
        test_build_activity{}.service(s1).shared(),
        {},
      },

      {
        "should allow insertion after strict sequence within middle position",
        Lock::Position::middle(),
        test_build_activity{}.service(s2).shared(),
        TerminalActivity,
        {},
      },
      {
        "should allow insertion after strict sequence before arrival within middle position",
        Lock::Position::middle(),
        test_build_activity{}.service(s2).shared(),
        OtherActivity,
        {},
      },
      //// Departure ////
      {
        "should deny insertion in between within departure position",
        Lock::Position::departure(),
        test_build_activity{}.service(s1).shared(),
        test_build_activity{}.service(s2).shared(),
        {{false, 3}},
      },
      {
        "should allow insertion after strict sequence within departure position",
        Lock::Position::departure(),
        test_build_activity{}.service(s2).shared(),
        OtherActivity,
        {},
      },
      {
        "should allow insertion after strict sequence before arrival within departure position",
        Lock::Position::departure(),
        test_build_activity{}.service(s2).shared(),
        TerminalActivity,
        {},
      },

      {
        "should deny insertion before strict sequence within departure position",
        Lock::Position::departure(),
        TerminalActivity,
        test_build_activity{}.service(s1).shared(),
        {{false, 3}},
      },

      //// Arrival ////
      {
        "should deny insertion in between within arrival position",
        Lock::Position::arrival(),
        test_build_activity{}.service(s1).shared(),
        test_build_activity{}.service(s2).shared(),
        {{false, 3}},
      },
      {
        "should allow insertion before strict sequence within arrival position",
        Lock::Position::arrival(),
        OtherActivity,
        test_build_activity{}.service(s1).shared(),
        {},
      },
      {
        "should allow insertion before strict sequence before departure within arrival position",
        Lock::Position::arrival(),
        TerminalActivity,
        test_build_activity{}.service(s1).shared(),
        {},
      },

      {
        "should deny insertion after strict sequence within arrival position",
        Lock::Position::arrival(),
        test_build_activity{}.service(s2).shared(),
        TerminalActivity,
        {{false, 3}},
      },

      //// Fixed ////
      {
        "should deny insertion in between within fixed position",
        Lock::Position::fixed(),
        test_build_activity{}.service(s1).shared(),
        test_build_activity{}.service(s2).shared(),
        {{false, 3}},
      },
      {
        "should deny insertion before strict sequence within fixed position",
        Lock::Position::fixed(),
        TerminalActivity,
        test_build_activity{}.service(s1).shared(),
        {{false, 3}},
      },
      {
        "should deny insertion after strict sequence within fixed position",
        Lock::Position::fixed(),
        test_build_activity{}.service(s2).shared(),
        TerminalActivity,
        {{false, 3}},
      },
    }));

  GIVEN("fleet with 1 vehicle and three jobs") {
    auto fleet = Fleet();
    fleet.add(test_build_driver{}.owned());
    fleet.add(test_build_vehicle{}.id("v1").details(asDetails(0, {}, {0, 100})).owned());
    auto registry = std::make_shared<Registry>(fleet);
    auto newJob = test_build_service{}.id("new").shared();
    auto actor = (registry->available() | to_vector).front();

    WHEN("activity level constraint called having job lock for one actor") {
      auto locks = std::vector<Lock>{Lock{[](const auto& a) { return true; },
                                          {Lock::Detail{Lock::Order::Strict, position, {as_job(s1), as_job(s2)}}}}};
      auto slnCtx = InsertionSolutionContext{{}, {}, {}, {}, registry};
      auto routeCtx = test_build_insertion_route_context{}.route(test_build_route{}.actor(actor).shared()).owned();
      auto actCtx = test_build_insertion_activity_context{}
                      .prev(prev)
                      .target(test_build_activity{}.service(newJob).shared())
                      .next(next)
                      .owned();
      auto constraint = ActorJobLock{locks};
      constraint.accept(slnCtx);

      THEN(message) {
        auto result = constraint.hard(routeCtx, actCtx);
        REQUIRE(result == expected);
      }
    }
  }
}
}