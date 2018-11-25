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

  vrp::models::common::Cost testActivityCosts(std::shared_ptr<InsertionRouteContext> routeCtx,
                                              std::shared_ptr<InsertionActivityContext> actCtx,
                                              const InsertionProgress& progress) const {
    return activityCosts(*routeCtx, *actCtx, progress);
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
      routeCtx->route->tour.add(prev).add(next);

      THEN("cost for vehicle is zero") {
        auto cost = evaluator.testVehicleCosts(routeCtx);

        REQUIRE(cost == 0);
      }
    }

    WHEN("using different actor returning to start") {
      auto [routeCtx, _] = differentActor(prev, target, next);
      routeCtx->route->tour.add(prev).add(next);

      THEN("cost for vehicle is correct") {
        auto cost = evaluator.testVehicleCosts(routeCtx);

        REQUIRE(cost == 20);
      }
    }

    WHEN("using different actor returning to different location") {
      auto [routeCtx, _] = differentActor(prev, target, next, 5);
      routeCtx->route->tour.add(prev).add(next);

      THEN("cost for vehicle is correct") {
        auto cost = evaluator.testVehicleCosts(routeCtx);

        REQUIRE(cost == 10);
      }
    }
  }
}

SCENARIO("job insertion evaluator estimates activity costs", "[algorithms][construction][insertion]") {
  auto evaluator = FakeJobInsertionEvaluator(std::make_shared<TestTransportCosts>(),  //
                                             std::make_shared<ActivityCosts>());

  GIVEN("empty tour") {
    // old: 0
    // new: d(10) + t(10 + 1)
    auto target = test_build_activity{}.duration(1).location(5);
    auto progress = test_build_insertion_progress{}.owned();

    WHEN("inserting new activity with the same actor") {
      auto [routeCtx, actCtx] = sameActor(target.shared());

      THEN("cost for activity is correct") {
        auto cost = evaluator.testActivityCosts(routeCtx, actCtx, progress);

        REQUIRE(cost == 21);
      }
    }

    WHEN("inserting new activity with different actor") {
      auto [routeCtx, actCtx] = differentActor(target.shared());

      THEN("cost for activity is correct") {
        auto cost = evaluator.testActivityCosts(routeCtx, actCtx, progress);

        REQUIRE(cost == 21);
      }
    }
  }

  GIVEN("tour with two activities") {
    auto progress = test_build_insertion_progress{}.owned();
    auto prev = test_build_activity{}.location(10).schedule({0, 10}).shared();
    auto target = test_build_activity{}.location(30).duration(10).shared();
    auto next = test_build_activity{}.location(20).time({40, 70}).shared();

    // old: d(10 + 20) + t(10 + 20 + 20) = 80
    // new: d(10 + 10 + 30) + t(20 + 10 + 30) = 110
    WHEN("inserting in between new activity with the same actor") {
      auto [routeCtx, actCtx] = sameActor(prev, target, next);
      routeCtx->route->tour.add(prev).add(next);

      THEN("cost for activity is correct") {
        auto cost = evaluator.testActivityCosts(routeCtx, actCtx, progress);

        REQUIRE(cost == 30);
      }
    }
  }
}
}
