#include "algorithms/construction/constraints/VehicleActivitySize.hpp"

#include "models/costs/ActivityCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "test_utils/algorithms/construction/Contexts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;
using namespace vrp::models::problem;
using namespace vrp::test;

namespace {

using Demand = VehicleActivitySize<int>::Demand;
const auto CurrentKey = VehicleActivitySize<int>::StateKeyCurrent;
const auto DemandKey = VehicleActivitySize<int>::DimKeyDemand;
const auto CapacityKey = VehicleActivitySize<int>::DimKeyCapacity;

Demand
createDemand(int size) {
  return size > 0 ? Demand{{size, 0}, {0, 0}} : Demand{{0, 0}, {std::abs(size), 0}};
}

InsertionRouteContext
createRouteState(const Fleet& fleet) {
  auto route = test_build_route{}.actor(getActor("v1", fleet)).shared();
  auto state = std::make_shared<InsertionRouteState>();
  return {route, state};
}

std::shared_ptr<Fleet>
createFleet() {
  auto fleet = std::make_shared<Fleet>();
  fleet->add(test_build_vehicle{}
               .dimens({{"id", std::string("v1")}, {CapacityKey, 10}})
               .details(asDetails(0, {}, {0, 100}))
               .owned());
  return fleet;
}

/// Adds activity with given id, departure, and demand.
Tour::Activity
activity(const std::string& id, Timestamp departure, int size) {
  return test_build_activity{}
    .schedule({0, departure})
    .service(test_build_service{}.dimens({{DemandKey, createDemand(size)}, {"id", id}}).shared())
    .shared();
}
}

namespace vrp::test {

SCENARIO("vehicle activity size calculates proper current load", "[algorithms][construction][constraints][size]") {
  GIVEN("fleet with 1 vehicle and route with service jobs") {
    auto fleet = createFleet();

    WHEN("accept route with three service activities") {
      auto routeState = createRouteState(*fleet);
      auto [route, state] = routeState;

      auto [s1, s2, s3, start, expS1, expS2, expS3, end] = GENERATE(table<int, int, int, int, int, int, int, int>({
        {-1, 2, -3, 4, 3, 5, 2, 2},  //
        {1, -2, 3, 2, 3, 1, 4, 4},
        {0, 1, 0, 0, 0, 1, 1, 1},
      }));

      route->tour.add(activity("s1", 1, s1)).add(activity("s2", 2, s2)).add(activity("s3", 3, s3));
      VehicleActivitySize<int>{}.accept(routeState);

      THEN("has correct load at start") { REQUIRE(state->get<int>(CurrentKey, route->start).value_or(-1) == start); }

      THEN("has correct load at end") { REQUIRE(state->get<int>(CurrentKey, route->end).value_or(-1) == end); }

      THEN("has correct current load at each activity") {
        REQUIRE(state->get<int>(CurrentKey, route->tour.get(0)).value_or(-1) == expS1);
        REQUIRE(state->get<int>(CurrentKey, route->tour.get(1)).value_or(-1) == expS2);
        REQUIRE(state->get<int>(CurrentKey, route->tour.get(2)).value_or(-1) == expS3);
      }
    }
  }
}

SCENARIO("vehicle activity size handles different results", "[algorithms][construction][constraints][size]") {
  GIVEN("fleet with 1 vehicle and route with service jobs") {
    auto fleet = createFleet();

    WHEN("check route and service job with different sizes") {
      auto routeState = createRouteState(*fleet);
      auto [route, state] = routeState;
      auto routeCtx = test_build_insertion_route_context{}.route(route).state(state).owned();

      auto [size, expected] = GENERATE(table<int, std::optional<int>>({{11, std::optional<int>{2}},  //
                                                                       {10, std::optional<int>{}}}));

      auto result = VehicleActivitySize<int>{}.hard(
        routeCtx,
        as_job(test_build_service{}.dimens({{DemandKey, createDemand(size)}, {"id", std::string("v1")}}).shared()));

      THEN("constraint check result is correct") { REQUIRE(result == expected); }
    }
  }
}

SCENARIO("vehicle activity size checks diffefrent cases", "[algorithms][construction][constraints][size]") {
  GIVEN("fleet with 1 vehicle and route with service jobs") {
    auto fleet = createFleet();
    WHEN("check route and service activity with different states") {
      auto [s1, s2, s3, expected] =
        GENERATE(table<int, int, int, HardActivityConstraint::Result>({                       //
                                                                       {1, 1, 1, success()},  //
                                                                       {1, 10, 1, stop(2)},
                                                                       {-5, -1, -5, stop(2)},
                                                                       {5, 1, 5, stop(2)},
                                                                       {-5, 1, 5, success()},
                                                                       {5, 1, -5, stop(2)},
                                                                       {4, -1, -5, success()}}));

      auto routeState = createRouteState(*fleet);
      auto [route, state] = routeState;
      route->tour.add(activity("s1", 1, s1)).add(activity("s3", 3, s3));
      auto sized = VehicleActivitySize<int>{};
      sized.accept(routeState);
      auto routeCtx = test_build_insertion_route_context{}.route(route).state(state).owned();
      auto actCtx =
        test_build_insertion_activity_context{}
          .prev(getActivity(routeCtx, 0))
          .target(
            test_build_activity{}
              .service(
                test_build_service{}.dimens({{"id", std::string("service")}, {DemandKey, createDemand(s2)}}).shared())
              .shared())
          .next(getActivity(routeCtx, 1))
          .owned();


      THEN("constraint check result is correct") {
        auto result = sized.hard(routeCtx, actCtx);

        REQUIRE(result == expected);
      }
    }
  }
}

SCENARIO("vehicle activity size handles tour with three services", "[algorithms][construction][constraints][size]") {
  GIVEN("fleet with 1 vehicle and route with service jobs") {
    auto fleet = createFleet();
    WHEN("has tree services with exact max size") {
      auto [prev, next] = GENERATE(table<int, int>({{-1, 0}, {0, 1}, {1, 2}, {2, -2}}));
      auto routeState = createRouteState(*fleet);
      auto [route, state] = routeState;
      route->tour.add(activity("s1", 1, -3)).add(activity("s2", 2, -5)).add(activity("s3", 3, -2));
      auto sized = VehicleActivitySize<int>{};
      sized.accept(routeState);
      auto routeCtx = test_build_insertion_route_context{}.route(route).state(state).owned();
      auto actCtx =
        test_build_insertion_activity_context{}
          .prev(getActivity(routeCtx, prev))
          .target(
            test_build_activity{}
              .service(
                test_build_service{}.dimens({{"id", std::string("service")}, {DemandKey, createDemand(-1)}}).shared())
              .shared())
          .next(getActivity(routeCtx, next))
          .owned();

      THEN("cannot insert new service") {
        auto result = sized.hard(routeCtx, actCtx);
        REQUIRE(result == stop(2));
      }
    }
  }
}
}
