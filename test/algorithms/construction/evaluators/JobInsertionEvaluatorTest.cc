#include "algorithms/construction/evaluators/JobInsertionEvaluator.hpp"

#include "models/costs/ActivityCosts.hpp"
#include "test_utils/algorithms/construction/Contexts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>
#include <utility>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;

namespace {
struct FakeJobInsertionEvaluator final : public JobInsertionEvaluator {
  explicit FakeJobInsertionEvaluator(std::shared_ptr<const TransportCosts> transportCosts,
                                     std::shared_ptr<const ActivityCosts> activityCosts) :
    JobInsertionEvaluator(std::move(transportCosts), std::move(activityCosts)) {}

  vrp::models::common::Cost testVehicleCosts(std::shared_ptr<InsertionRouteContext> routeCtx) const {
    return vehicleCosts(*routeCtx);
  }
};
}

namespace vrp::test {

SCENARIO("job insertion evaluator estimates vehicle costs", "[algorithms][construction][insertion]") {
  auto evaluator = FakeJobInsertionEvaluator(std::make_shared<TestTransportCosts>(),  //
                                             std::make_shared<ActivityCosts>());

  GIVEN("empty tour") {
    auto target = test_build_activity{}.location(5);

    WHEN("using the same actor") {
      auto [routeCtx, _] = sameActor(target.shared());

      THEN("cost for vehicle is zero") {
        auto cost = evaluator.testVehicleCosts(routeCtx);

        REQUIRE(cost == 0);
      }
    }

    WHEN("using different actor") {
      auto [routeCtx, _] = differentActor(target.shared());

      THEN("cost for vehicle is zero") {
        auto cost = evaluator.testVehicleCosts(routeCtx);

        REQUIRE(cost == 0);
      }
    }
  }

  GIVEN("tour with two activities") {
    auto prev = test_build_activity{}.location(5).schedule({0, 5}).shared();
    auto target = test_build_activity{}.location(10).duration(1).shared();
    auto next = test_build_activity{}.location(15).shared();

    WHEN("using the same actor") {
      auto [routeCtx, _] = sameActor(prev, target, next);
      routeCtx->route.first->tour.add(prev).add(next);

      THEN("cost for vehicle is zero") {
        auto cost = evaluator.testVehicleCosts(routeCtx);

        REQUIRE(cost == 0);
      }
    }

    WHEN("using different actor returning to start") {
      auto [routeCtx, _] = differentActor(prev, target, next);
      routeCtx->route.first->tour.add(prev).add(next);

      THEN("cost for vehicle is correct") {
        auto cost = evaluator.testVehicleCosts(routeCtx);

        REQUIRE(cost == 20);
      }
    }

    WHEN("using different actor returning to different location") {
      auto [routeCtx, _] = differentActor(prev, target, next, 5);
      routeCtx->route.first->tour.add(prev).add(next);

      THEN("cost for vehicle is correct") {
        auto cost = evaluator.testVehicleCosts(routeCtx);

        REQUIRE(cost == 10);
      }
    }
  }
}
}
