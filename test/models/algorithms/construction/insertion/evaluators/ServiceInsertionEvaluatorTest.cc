#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"
#include "models/costs/ActivityCosts.hpp"

#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::costs;

namespace vrp::test {

SCENARIO("service insertion evaluator", "[algorithms][construction][insertion]") {
  GIVEN("empty tour") {
    auto progress = test_build_insertion_progress{}.owned();
    auto constraint = std::make_shared<InsertionConstraint>();
    auto route = test_build_route{}.owned();
    auto evaluator = ServiceInsertionEvaluator(std::make_shared<TestTransportCosts>(),
                                               std::make_shared<ActivityCosts>(),
                                               constraint);

    WHEN("service has failed constraint") {
      constraint->addHardRoute([](const auto &, const auto &) { return InsertionConstraint::HardRouteResult{42}; });
      auto result = evaluator.evaluate(ranges::get<0>(DefaultService),
                                       test_build_insertion_route_context{}.owned(),
                                       progress);

      THEN("returns insertion failure with proper code") {
        REQUIRE (result.index() == 1);
        REQUIRE (ranges::get<1>(result).constraint == 42);
      }
    }

    WHEN("service is ok") {
      auto result = evaluator.evaluate(ranges::get<0>(DefaultService),
                                       test_build_insertion_route_context{}.owned(),
                                       progress);

      THEN("returns insertion success") {
        REQUIRE (result.index() == 0);
        REQUIRE (ranges::get<0>(result).index == 0);
        REQUIRE (ranges::get<0>(result).departure == 0);
        REQUIRE (ranges::get<0>(result).activity->location == DefaultJobLocation);
      }
    }

    WHEN("service has no location") {
      auto service = test_build_service{}.details({{{}, DefaultDuration, {DefaultTimeWindow}}}).shared();
      auto result = evaluator.evaluate(service, test_build_insertion_route_context{}.owned(), progress);

      THEN("returns correct insertion success") {
        REQUIRE (result.index() == 0);
        REQUIRE (ranges::get<0>(result).index == 0);
        REQUIRE (ranges::get<0>(result).departure == 0);
        REQUIRE (ranges::get<0>(result).activity->location == 0);
      }
    }
  }

  GIVEN("tour with two simple activities without time windows") {
    auto prev = test_build_activity{}.location(5).duration(0).schedule({5, 5}).shared();
    auto next = test_build_activity{}.location(10).schedule({10, 10}).duration(0).shared();

    auto constraint = std::make_shared<InsertionConstraint>();
    auto progress = test_build_insertion_progress{}.owned();
    auto routeCtx = test_build_insertion_route_context{}.shared();
    routeCtx->route->tour.add(prev).add(next);

    auto evaluator = ServiceInsertionEvaluator(std::make_shared<TestTransportCosts>(),
                                               std::make_shared<ActivityCosts>(),
                                               constraint);

    auto[location, index] = GENERATE(std::make_tuple(3, 0), std::make_tuple(8, 1));

    WHEN("service is inserted") {
      auto service = test_build_service{}.details({{{location}, 0, {DefaultTimeWindow}}}).shared();
      auto result = evaluator.evaluate(service, *routeCtx, progress);

      THEN("returns correct insertion success") {
        REQUIRE (result.index() == 0);
        REQUIRE (ranges::get<0>(result).departure == 0);
        REQUIRE (ranges::get<0>(result).index == index);
        REQUIRE (ranges::get<0>(result).activity->location == location);
      }
    }
  }
}

}