#include "algorithms/construction/insertion/evaluators/JobInsertionEvaluator.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestActivityCosts.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;

namespace {
struct FakeJobInsertionEvaluator final : public JobInsertionEvaluator {
  explicit FakeJobInsertionEvaluator(std::shared_ptr<const TransportCosts> transportCosts,
                                     std::shared_ptr<const ActivityCosts> activityCosts) :
    JobInsertionEvaluator(std::move(transportCosts), std::move(activityCosts)) {}

  vrp::models::common::Cost testExtraCosts(const InsertionRouteContext& routeCtx,
                                           const InsertionActivityContext& actCtx,
                                           const InsertionProgress& progress) const {
    return extraCosts(routeCtx, actCtx, progress);
  }
};
}

namespace vrp::test {

SCENARIO("job insertion evaluator", "[algorithms][construction][insertion]") {
  GIVEN("empty tour") {
    auto evaluator = FakeJobInsertionEvaluator(std::make_shared<TestTransportCosts>(),  //
                                               std::make_shared<TestActivityCosts>());
    auto target = test_build_activity{}.withLocation(5);

    WHEN("inserting new activity with the same actor") {
      auto progress = test_build_insertion_progress{}.owned();
      auto routeCtx = test_build_insertion_route_context{}.owned();
      auto actCtx = test_build_insertion_activity_context{}  //
                      .withPrev(routeCtx.route->start)
                      .withTarget(target.shared())
                      .withNext(routeCtx.route->end)
                      .owned();

      THEN("extra cost for activity is correct") {
        auto cost = evaluator.testExtraCosts(routeCtx, actCtx, progress);

        REQUIRE(cost == 10);
      }
    }
  }
}
}