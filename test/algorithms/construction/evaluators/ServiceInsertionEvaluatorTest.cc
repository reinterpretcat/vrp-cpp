#include "algorithms/construction/evaluators/ServiceInsertionEvaluator.hpp"

#include "algorithms/construction/constraints/VehicleActivityTiming.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>
#include <tuple>
#include <vector>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;

using namespace ranges;

namespace {

auto
createContext(const Tour::Activity& prev, const Tour::Activity& next) {
  using namespace vrp::test;

  auto transport = std::make_shared<TestTransportCosts>();
  auto activity = std::make_shared<ActivityCosts>();
  auto fleet = std::make_shared<Fleet>();
  auto constraint = std::make_shared<InsertionConstraint>();

  (*fleet).add(*DefaultDriver).add(*DefaultVehicle);
  constraint->add<VehicleActivityTiming>(std::make_shared<VehicleActivityTiming>(fleet, transport, activity));


  auto routeCtx = test_build_insertion_route_context{}.add(prev).add(next).shared();
  auto evaluator = std::make_shared<ServiceInsertionEvaluator>();

  return std::tuple<std::shared_ptr<InsertionRouteContext>,
                    std::shared_ptr<ServiceInsertionEvaluator>,
                    std::shared_ptr<InsertionConstraint>>{routeCtx, evaluator, constraint};
}

std::vector<TimeWindow>
times(std::initializer_list<TimeWindow> tws) {
  return tws;
}

std::vector<Location>
locations(std::initializer_list<Location> locs) {
  return locs;
}

std::vector<Service::Detail>
details(std::initializer_list<Service::Detail> ds) {
  return ds;
}
}

namespace vrp::test {

SCENARIO("service insertion evaluator", "[algorithms][construction][insertion]") {
  GIVEN("empty tour") {
    auto progress = test_build_insertion_progress{}.owned();
    auto constraint = std::make_shared<InsertionConstraint>();
    auto route = test_build_route{}.owned();
    auto evaluator = ServiceInsertionEvaluator{};

    WHEN("service is ok") {
      auto result = evaluator.evaluate(
        ranges::get<0>(DefaultService), test_build_insertion_route_context{}.owned(), *constraint, progress);

      THEN("returns insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).activities[0].second == 0);
        REQUIRE(ranges::get<0>(result).activities[0].first->detail.location == DefaultJobLocation);
      }
    }

    WHEN("service has no location") {
      auto service = test_build_service{}.details({{{}, DefaultDuration, {DefaultTimeWindow}}}).shared();
      auto result = evaluator.evaluate(service, test_build_insertion_route_context{}.owned(), *constraint, progress);

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).activities[0].second == 0);
        REQUIRE(ranges::get<0>(result).activities[0].first->detail.location == 0);
      }
    }
  }

  GIVEN("tour with two simple activities and service with time windows variations") {
    auto prev = test_build_activity{}.location(5).schedule({5, 5}).shared();
    auto next = test_build_activity{}.location(10).schedule({10, 10}).shared();
    auto [routeCtx, evaluator, constraint] = createContext(prev, next);

    auto [location, tws, index] = GENERATE(std::make_tuple(3, times({DefaultTimeWindow}), 0),
                                           std::make_tuple(8, times({DefaultTimeWindow}), 1),
                                           std::make_tuple(7, times({{15, 20}}), 2),
                                           std::make_tuple(7, times({{15, 20}, {7, 8}}), 1));

    WHEN("service is inserted") {
      auto service = test_build_service{}.details({{{location}, 0, tws}}).shared();
      auto result = evaluator->evaluate(service, *routeCtx, *constraint, test_build_insertion_progress{}.owned());

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).activities[0].second == index);
        REQUIRE(ranges::get<0>(result).activities[0].first->detail.location == location);
      }
    }
  }

  GIVEN("tour with two simple activities and different locations") {
    auto prev = test_build_activity{}.location(5).schedule({5, 5}).shared();
    auto next = test_build_activity{}.location(10).schedule({10, 10}).shared();
    auto [routeCtx, evaluator, constraint] = createContext(prev, next);

    auto [locs, index, loc] =
      GENERATE(std::make_tuple(locations({3}), 0, 3), std::make_tuple(locations({20, 3}), 0, 3));

    WHEN("service is inserted") {
      auto ds = view::all(locs) | view::transform([&](const auto& l) {
                  return Service::Detail{{l}, 0, {DefaultTimeWindow}};
                });
      auto service = test_build_service{}.details(std::vector<Service::Detail>{ds}).shared();
      auto result = evaluator->evaluate(service, *routeCtx, *constraint, test_build_insertion_progress{}.owned());

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).activities[0].second == index);
        REQUIRE(ranges::get<0>(result).activities[0].first->detail.location == loc);
      }
    }
  }

  GIVEN("tour with two simple activities and different locations, including constraints") {
    auto prev = test_build_activity{}.location(5).schedule({5, 5}).shared();
    auto next = test_build_activity{}.location(10).schedule({10, 10}).shared();
    auto [routeCtx, evaluator, constraint] = createContext(prev, next);

    auto [ds, index, loc] = GENERATE(
      std::make_tuple(details({{{3}, 0, {DefaultTimeWindow}}}), 0, 3),
      std::make_tuple(details({{{20}, 0, {DefaultTimeWindow}}, {{3}, 0, times({{0, 2}})}}), 1, 20),
      std::make_tuple(details({{{12}, 0, {DefaultTimeWindow}}, {{11}, 0, times({DefaultTimeWindow})}}), 1, 11));

    WHEN("service is inserted") {
      auto service = test_build_service{}.details(std::vector<Service::Detail>{ds}).shared();
      auto result = evaluator->evaluate(service, *routeCtx, *constraint, test_build_insertion_progress{}.owned());

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).activities[0].second == index);
        REQUIRE(ranges::get<0>(result).activities[0].first->detail.location == loc);
      }
    }
  }
}
}