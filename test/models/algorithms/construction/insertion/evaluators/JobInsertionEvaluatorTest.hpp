#include "algorithms/construction/insertion/evaluators/JobInsertionEvaluator.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestActivityCosts.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;

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

SCENARIO("job insertion evaluator", "[algorithms][insertion]") {
  GIVEN("empty tour") {
    auto evaluator = FakeJobInsertionEvaluator(std::make_shared<TestTransportCosts>(),  //
                                               std::make_shared<TestActivityCosts>());

    auto activity1 = test_build_activity{}.withLocation(0).shared();
    auto activity2 = test_build_activity{}.withLocation(10).shared();

    WHEN("inserting new activity with the same actor") {
      auto progress = test_build_insertion_progress{}.owned();
      auto routeCtx = test_build_insertion_route_context{}.owned();
      auto actCtx = test_build_insertion_activity_context{}  //
                      .withPrev(activity1)
                      .withTarget(activity2)
                      .withNext(activity1)
                      .owned();


      THEN("extra cost for activity is correct") {
        auto cost = evaluator.testExtraCosts(routeCtx, actCtx, progress);

        REQUIRE(cost == 20);
      }
    }
  }
}
}