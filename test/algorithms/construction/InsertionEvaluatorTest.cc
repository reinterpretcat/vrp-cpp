#include "algorithms/construction/InsertionEvaluator.hpp"

#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/construction/constraints/VehicleActivityTiming.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::algorithms::construction;
using namespace vrp::test;
using namespace ranges;

namespace {

std::shared_ptr<Fleet>
createFleet() {
  auto fleet = std::make_shared<Fleet>();
  (*fleet).add(test_build_driver{}.owned()).add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned());
  return fleet;
}

Tour::Activity
createActivity(const std::string& id, int size) {
  return test_build_activity{}.job(as_job(test_build_service{}.id(id).dimens({{"size", size}}).shared())).shared();
}

auto
createContext(const Tour::Activity& prev, const Tour::Activity& next) {
  using namespace vrp::test;

  auto constraint = std::make_shared<InsertionConstraint>();
  constraint->add<VehicleActivityTiming>(std::make_shared<VehicleActivityTiming>(
    createFleet(), std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>()));

  return test_build_insertion_context{}
    .constraint(constraint)
    .progress(test_build_insertion_progress{}.owned())
    .routes({test_build_insertion_route_context{}.add(prev).add(next).owned()})
    .registry(std::make_shared<Registry>(*createFleet()))
    .owned();
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

SCENARIO("insertion evaluator can insert service", "[algorithms][construction][insertion]") {
  GIVEN("empty tour") {
    auto route = test_build_route{}.owned();

    auto context = test_build_insertion_context{}
                     .constraint(std::make_shared<InsertionConstraint>())
                     .progress(test_build_insertion_progress{}.owned())
                     .routes({test_build_insertion_route_context{}.owned()})
                     .registry(std::make_shared<Registry>(*createFleet()))
                     .owned();

    WHEN("service is ok") {
      auto result = InsertionEvaluator{}.evaluate(DefaultService, context);

      THEN("returns insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).activities[0].second == 0);
        REQUIRE(ranges::get<0>(result).activities[0].first->detail.location == DefaultJobLocation);
      }
    }

    WHEN("service has no location") {
      auto result = InsertionEvaluator{}.evaluate(
        as_job(test_build_service{}.details({{{}, DefaultDuration, {DefaultTimeWindow}}}).shared()), context);

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
    auto context = createContext(prev, next);

    auto [location, tws, index] = GENERATE(std::make_tuple(3, times({DefaultTimeWindow}), 0),
                                           std::make_tuple(8, times({DefaultTimeWindow}), 1),
                                           std::make_tuple(7, times({{15, 20}}), 2),
                                           std::make_tuple(7, times({{15, 20}, {7, 8}}), 1));

    WHEN("service is inserted") {
      auto service = test_build_service{}.details({{{location}, 0, tws}}).shared();
      auto result =
        InsertionEvaluator{}.evaluate(as_job(test_build_service{}.details({{{location}, 0, tws}}).shared()), context);

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
    auto context = createContext(prev, next);

    auto [locs, index, loc] =
      GENERATE(std::make_tuple(locations({3}), 0, 3), std::make_tuple(locations({20, 3}), 0, 3));

    WHEN("service is inserted") {
      auto ds = view::all(locs) | view::transform([&](const auto& l) {
                  return Service::Detail{{l}, 0, {DefaultTimeWindow}};
                });

      auto result = InsertionEvaluator{}.evaluate(
        as_job(test_build_service{}.details(std::vector<Service::Detail>{ds}).shared()), context);

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
    auto context = createContext(prev, next);

    auto [ds, index, loc] = GENERATE(
      std::make_tuple(details({{{3}, 0, {DefaultTimeWindow}}}), 0, 3),
      std::make_tuple(details({{{20}, 0, {DefaultTimeWindow}}, {{3}, 0, times({{0, 2}})}}), 1, 20),
      std::make_tuple(details({{{12}, 0, {DefaultTimeWindow}}, {{11}, 0, times({DefaultTimeWindow})}}), 1, 11));

    WHEN("service is inserted") {
      auto result = InsertionEvaluator{}.evaluate(
        as_job(test_build_service{}.details(std::vector<Service::Detail>{ds}).shared()), context);

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).activities[0].second == index);
        REQUIRE(ranges::get<0>(result).activities[0].first->detail.location == loc);
      }
    }
  }
}

SCENARIO("insertion evaluator can handle service insertion with time constraints",
         "[algorithms][construction][insertion]") {
  using EndLoc = std::optional<Location>;

  auto [s1, v1, v2, used, cost] = GENERATE(table<Location, EndLoc, EndLoc, std::string, Cost>({
    {3, {}, {}, "v1", (3 + 3) * 2},
    {27, {}, {}, "v2", (7 + 7) * 2},
    {11, {12}, {}, "v1", (12 + 12)},
  }));

  GIVEN("two different vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, v1, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{20, v2, {0, 100}}}).owned());
    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->add<VehicleActivityTiming>(std::make_shared<VehicleActivityTiming>(
      fleet, std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>()));

    auto evaluator = InsertionEvaluator{};

    WHEN("evaluates service insertion close to best vehicle") {
      auto service = test_build_service{}.location(s1).shared();

      auto result = evaluator.evaluate(
        as_job(service),
        test_build_insertion_context{}.constraint(constraint).registry(std::make_shared<Registry>(*fleet)).owned());

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(get_vehicle_id{}(*ranges::get<0>(result).context.route->actor->vehicle) == used);
        REQUIRE(ranges::get<0>(result).cost == cost);
      }
    }
  }
}

SCENARIO("insertion evaluator can handle service insertion with violation", "[algorithms][construction][insertion]") {
  GIVEN("failed constraint") {
    auto fleet = createFleet();
    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->addHardRoute([](const auto&, const auto&) { return HardRouteConstraint::Result{42}; });

    WHEN("service is evaluated") {
      auto result = InsertionEvaluator{}.evaluate(
        DefaultService,
        test_build_insertion_context{}.constraint(constraint).registry(std::make_shared<Registry>(*fleet)).owned());

      THEN("returns insertion failure with proper code") {
        REQUIRE(result.index() == 1);
        REQUIRE(ranges::get<1>(result).constraint == 42);
      }
    }
  }
}

SCENARIO("insertion evaluator can insert sequence without constraints", "[algorithms][construction][insertion]") {
  GIVEN("empty tour") {
    auto route = test_build_route{}.owned();

    auto context = test_build_insertion_context{}
                     .constraint(std::make_shared<InsertionConstraint>())
                     .progress(test_build_insertion_progress{}.owned())
                     .routes({test_build_insertion_route_context{}.owned()})
                     .registry(std::make_shared<Registry>(*createFleet()))
                     .owned();

    WHEN("sequence is ok") {
      THEN("returns insertion success with two activities") {
        auto result = InsertionEvaluator{}.evaluate(as_job(test_build_sequence{}
                                                             .id("sequence")
                                                             .service(test_build_service{}.id("s1").owned())
                                                             .service(test_build_service{}.id("s2").owned())
                                                             .shared()),
                                                    context);
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).activities.size() == 2);
        REQUIRE(ranges::get<0>(result).activities[0].second == 0);
        REQUIRE(ranges::get<0>(result).activities[1].second == 1);
      }
    }
  }
}
}
