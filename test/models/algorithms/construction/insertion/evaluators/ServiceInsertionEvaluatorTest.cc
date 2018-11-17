#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"

#include "models/costs/ActivityCosts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>
#include <vector>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;

namespace {

auto
createContext(const Tour::Activity& prev, const Tour::Activity& next) {
  auto routeCtx = vrp::test::test_build_insertion_route_context{}.add(prev).add(next).shared();
  auto evaluator = std::make_shared<ServiceInsertionEvaluator>(std::make_shared<vrp::test::TestTransportCosts>(),
                                                               std::make_shared<ActivityCosts>(),
                                                               std::make_shared<InsertionConstraint>());

  return std::pair<std::shared_ptr<InsertionRouteContext>, std::shared_ptr<ServiceInsertionEvaluator>>{routeCtx,
                                                                                                       evaluator};
}

std::vector<TimeWindow>
times(std::initializer_list<TimeWindow> tws) {
  return tws;
}
}

namespace vrp::test {

SCENARIO("service insertion evaluator", "[algorithms][construction][insertion]") {
  GIVEN("empty tour") {
    auto progress = test_build_insertion_progress{}.owned();
    auto constraint = std::make_shared<InsertionConstraint>();
    auto route = test_build_route{}.owned();
    auto evaluator =
      ServiceInsertionEvaluator(std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>(), constraint);

    WHEN("service has failed constraint") {
      constraint->addHardRoute([](const auto&, const auto&) { return InsertionConstraint::HardRouteResult{42}; });
      auto result =
        evaluator.evaluate(ranges::get<0>(DefaultService), test_build_insertion_route_context{}.owned(), progress);

      THEN("returns insertion failure with proper code") {
        REQUIRE(result.index() == 1);
        REQUIRE(ranges::get<1>(result).constraint == 42);
      }
    }

    WHEN("service is ok") {
      auto result =
        evaluator.evaluate(ranges::get<0>(DefaultService), test_build_insertion_route_context{}.owned(), progress);

      THEN("returns insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).index == 0);
        REQUIRE(ranges::get<0>(result).departure == 0);
        REQUIRE(ranges::get<0>(result).activity->location == DefaultJobLocation);
      }
    }

    WHEN("service has no location") {
      auto service = test_build_service{}.details({{{}, DefaultDuration, {DefaultTimeWindow}}}).shared();
      auto result = evaluator.evaluate(service, test_build_insertion_route_context{}.owned(), progress);

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).index == 0);
        REQUIRE(ranges::get<0>(result).departure == 0);
        REQUIRE(ranges::get<0>(result).activity->location == 0);
      }
    }
  }

  GIVEN("tour with two simple activities") {
    auto prev = test_build_activity{}.location(5).duration(0).schedule({5, 5}).shared();
    auto next = test_build_activity{}.location(10).schedule({10, 10}).duration(0).shared();
    auto [routeCtx, evaluator] = createContext(prev, next);

    auto [location, tws, index] = GENERATE(std::make_tuple(3, times({DefaultTimeWindow}), 0),
                                           std::make_tuple(8, times({DefaultTimeWindow}), 1),
                                           std::make_tuple(7, times({TimeWindow{15, 20}}), 2));

    WHEN("service is inserted") {
      auto service = test_build_service{}.details({{{location}, 0, tws}}).shared();
      auto result = evaluator->evaluate(service, *routeCtx, test_build_insertion_progress{}.owned());

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).departure == 0);
        REQUIRE(ranges::get<0>(result).index == index);
        REQUIRE(ranges::get<0>(result).activity->location == location);
      }
    }
  }
}
}