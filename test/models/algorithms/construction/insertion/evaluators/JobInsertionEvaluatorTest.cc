#include "algorithms/construction/insertion/evaluators/JobInsertionEvaluator.hpp"
#include "models/costs/ActivityCosts.hpp"

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

  vrp::models::common::Cost testExtraCosts(std::shared_ptr<InsertionRouteContext> routeCtx,
                                           std::shared_ptr<InsertionActivityContext> actCtx,
                                           const InsertionProgress& progress) const {
    return extraCosts(*routeCtx, *actCtx, progress);
  }
};

using TestContext = std::pair<std::shared_ptr<InsertionRouteContext>, std::shared_ptr<InsertionActivityContext>>;

TestContext sameActor(const vrp::models::solution::Tour::Activity& activity) {
  auto routeCtx = vrp::test::test_build_insertion_route_context{}.shared();
  auto actCtx = vrp::test::test_build_insertion_activity_context{}  //
      .prev(routeCtx->route->start)
      .target(activity)
      .next(routeCtx->route->end)
      .shared();
  return { routeCtx, actCtx };
}

TestContext differentActor(const vrp::models::solution::Tour::Activity& activity) {
  auto routeCtx = vrp::test::test_build_insertion_route_context{}
      .actor(vrp::test::test_build_actor{}
                     .vehicle(vrp::test::test_build_vehicle{}.start(20).shared())
                     .shared())
      .shared();
  auto actCtx = vrp::test::test_build_insertion_activity_context{}  //
      .prev(routeCtx->route->start)
      .target(activity)
      .next(routeCtx->route->end)
      .shared();
  return { routeCtx, actCtx };
}

}

namespace vrp::test {

SCENARIO("job insertion evaluator", "[algorithms][construction][insertion]") {
  auto evaluator = FakeJobInsertionEvaluator(std::make_shared<TestTransportCosts>(),  //
                                             std::make_shared<ActivityCosts>());

  GIVEN("empty tour") {
    // old costs: 0
    // new distance: 10
    // new time: driving: 10, service: 1, waiting: 0
    auto target = test_build_activity{}.location(5);
    auto progress = test_build_insertion_progress{}.owned();

    WHEN("inserting new activity with the same actor") {

      auto [routeCtx, actCtx] = sameActor(target.shared());

      THEN("extra cost for activity is correct") {
        auto cost = evaluator.testExtraCosts(routeCtx, actCtx, progress);

        REQUIRE(cost == 21);
      }
    }

    WHEN("inserting new activity with different actor") {
      auto [routeCtx, actCtx] = differentActor(target.shared());

      THEN("extra cost for activity is correct") {
        auto cost = evaluator.testExtraCosts(routeCtx, actCtx, progress);

        REQUIRE(cost == 21);
      }
    }
  }

  GIVEN("non empty tour") {
    WHEN("") {
//      auto prev = test_build_activity{}.withLocation(10);
//      auto target = test_build_activity{}.withLocation(60);
//      auto next = test_build_activity{}.withLocation(30).withDuration(10);
    }
  }
}
}