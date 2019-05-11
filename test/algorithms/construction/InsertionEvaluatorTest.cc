#include "algorithms/construction/InsertionEvaluator.hpp"

#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "test_utils/algorithms/construction/Factories.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::algorithms::construction;
using namespace vrp::test;
using namespace ranges;

namespace {

using InsertionData = std::pair<size_t, Location>;

std::shared_ptr<Fleet>
createFleet() {
  auto fleet = std::make_shared<Fleet>();
  (*fleet).add(test_build_driver{}.owned()).add(test_build_vehicle{}.id("v1").details({{0, {0}, {0, 100}}}).owned());
  return fleet;
}

Tour::Activity
createActivity(const std::string& id, int size) {
  return test_build_activity{}.service(test_build_service{}.id(id).dimens({{"size", size}}).shared()).shared();
}

std::shared_ptr<Problem>
createProblem(const std::shared_ptr<Fleet>& fleet) {
  auto activity = std::make_shared<ActivityCosts>();
  auto transport = std::make_shared<TestTransportCosts>();
  auto constraint = std::make_shared<InsertionConstraint>();

  constraint->add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, transport, activity, 1));
  return std::make_shared<Problem>(
    Problem{{}, {}, std::make_shared<std::vector<vrp::models::Lock>>(), constraint, {}, activity, transport, {}});
}

std::shared_ptr<Problem>
createProblem(std::shared_ptr<InsertionConstraint> constraint) {
  return std::make_shared<Problem>(Problem{{},
                                           {},
                                           std::make_shared<std::vector<vrp::models::Lock>>(),
                                           constraint,
                                           {},
                                           std::make_shared<ActivityCosts>(),
                                           std::make_shared<TestTransportCosts>(),
                                           {}});
}

auto
createContext(const Tour::Activity& prev, const Tour::Activity& next) {
  using namespace vrp::test;

  auto constraint = std::make_shared<InsertionConstraint>();
  constraint->add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(
    createFleet(), std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>(), 1));

  return test_build_insertion_context{}
    .problem(createProblem(constraint))
    .progress(test_build_insertion_progress{}.owned())
    .solution(build_insertion_solution_context{}
                .routes({test_build_insertion_route_context{}.insert(prev, 1).insert(next, 2).owned()})
                .registry(std::make_shared<Registry>(*createFleet()))
                .shared())
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

void
assertActivities(const InsertionSuccess& result, const std::vector<std::pair<size_t, Location>>& expected) {
  REQUIRE(result.activities.size() == expected.size());
  ranges::for_each(view::iota(0, expected.size()), [&](auto i) {
    REQUIRE(result.activities[i].second == expected[i].first);
    REQUIRE(result.activities[i].first->detail.location == expected[i].second);
  });
}
}

namespace vrp::test {

// region Service

SCENARIO("insertion evaluator can insert service", "[algorithms][construction][insertion][service]") {
  // TODO split tests
  GIVEN("empty tour") {
    auto context = test_build_insertion_context{}
                     .problem(createProblem(std::make_shared<InsertionConstraint>()))
                     .progress(test_build_insertion_progress{}.owned())
                     .solution(build_insertion_solution_context{}
                                 .routes({test_build_insertion_route_context{}.owned()})
                                 .registry(std::make_shared<Registry>(*createFleet()))
                                 .shared())

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
         "[algorithms][construction][insertion][service]") {
  using EndLoc = std::optional<Location>;

  auto [s1, v1, v2, used, cost] = GENERATE(table<Location, EndLoc, EndLoc, std::string, Cost>(
    {{3, {0}, {20}, "v1", (3 + 3) * 2}, {27, {0}, {20}, "v2", (7 + 7) * 2}, {11, {12}, {20}, "v1", (12 + 12)}}));

  GIVEN("two different vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, v1, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{20, v2, {0, 100}}}).owned());
    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(
      fleet, std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>(), 1));

    auto evaluator = InsertionEvaluator{};

    WHEN("evaluates service insertion close to best vehicle") {
      auto service = test_build_service{}.location(s1).shared();

      auto result = evaluator.evaluate(
        as_job(service),
        test_build_insertion_context{}
          .problem(createProblem(constraint))
          .solution(build_insertion_solution_context{}.registry(std::make_shared<Registry>(*fleet)).shared())
          .owned());

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(get_vehicle_id{}(*ranges::get<0>(result).context.route->actor->vehicle) == used);
        REQUIRE(ranges::get<0>(result).cost == cost);
      }
    }
  }
}

SCENARIO("insertion evaluator can handle service insertion with violation",
         "[algorithms][construction][insertion][service]") {
  GIVEN("failed constraint") {
    auto fleet = createFleet();
    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->addHardRoute(
      std::make_shared<HardRouteWrapper>([](const auto&, const auto&) { return HardRouteConstraint::Result{42}; }));

    WHEN("service is evaluated") {
      auto result = InsertionEvaluator{}.evaluate(DefaultService,
                                                  test_build_insertion_context{}
                                                    .problem(createProblem(constraint))
                                                    .solution(build_insertion_solution_context{}
                                                                .routes({test_build_insertion_route_context{}.owned()})
                                                                .registry(std::make_shared<Registry>(*fleet))
                                                                .shared())
                                                    .owned());

      THEN("returns insertion failure with proper code") {
        REQUIRE(result.index() == 1);
        REQUIRE(ranges::get<1>(result).constraint == 42);
      }
    }
  }
}

// endregion

// region Sequence

SCENARIO("insertion evaluator can insert sequence in empty tour", "[algorithms][construction][insertion][sequence]") {
  auto [s1, s2, cost, r1, r2] =
    GENERATE(table<Location, Location, Cost, std::pair<size_t, Location>, std::pair<size_t, Location>>({
      {3, 7, 28, {0, 3}, {1, 7}},
    }));

  GIVEN("empty tour and timing constraint") {
    auto fleet = createFleet();
    WHEN("sequence is inserted with activities with relaxed tw") {
      THEN("returns insertion success with two activities and proper cost") {
        auto result =
          InsertionEvaluator{}.evaluate(as_job(test_build_sequence{}
                                                 .id("sequence")
                                                 .service(test_build_service{}.id("s1").location(s1).shared())
                                                 .service(test_build_service{}.id("s2").location(s2).shared())
                                                 .shared()),
                                        test_build_insertion_context{}
                                          .problem(createProblem(fleet))
                                          .progress(test_build_insertion_progress{}.owned())
                                          .solution(build_insertion_solution_context{}
                                                      .routes({test_build_insertion_route_context{}.owned()})
                                                      .registry(std::make_shared<Registry>(*fleet))
                                                      .shared())

                                          .owned());
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).cost == cost);
        assertActivities(ranges::get<0>(result), {r1, r2});
      }
    }
  }
}

SCENARIO("insertion evaluator can insert sequence in tour with one activity",
         "[algorithms][construction][insertion][sequence]") {
  auto [exst, schedule, s1, s2, cost, r1, r2] =
    GENERATE(table<Location, Timestamp, Location, Location, Cost, InsertionData, InsertionData>({
      {5, 5, 3, 7, 8, {0, 3}, {1, 7}},  // s 3 [5] 7 e
      {5, 5, 7, 3, 8, {0, 7}, {2, 3}},  // s 7 [5] 3 e
    }));

  GIVEN("tour with one activity and timing constraint") {
    auto fleet = createFleet();
    auto context =
      test_build_insertion_context{}
        .problem(createProblem(fleet))
        .progress(test_build_insertion_progress{}.owned())
        .solution(build_insertion_solution_context{}
                    .routes({test_build_insertion_route_context{}
                               .insert(test_build_activity{}.location(exst).schedule({schedule, schedule}).shared(), 1)
                               .owned()})

                    .registry(std::make_shared<Registry>(*fleet))
                    .shared())
        .owned();
    auto rs = *context.solution->routes.begin();
    context.problem->constraint->accept(rs);

    WHEN("sequence is inserted with activities with relaxed tw") {
      THEN("returns insertion success with two activities and proper cost") {
        auto result =
          InsertionEvaluator{}.evaluate(as_job(test_build_sequence{}
                                                 .id("sequence")
                                                 .service(test_build_service{}.id("s1").location(s1).shared())
                                                 .service(test_build_service{}.id("s2").location(s2).shared())
                                                 .shared()),
                                        context);
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).cost == cost);
        assertActivities(ranges::get<0>(result), {r1, r2});
      }
    }
  }
}

SCENARIO("insertion evaluator can insert sequence with three activities in tour with two activities",
         "[algorithms][construction][insertion][sequence]") {
  auto [loc1, loc2, sh1, sh2, s1, s2, s3, cost, r1, r2, r3] = GENERATE(table<Location,
                                                                             Location,
                                                                             Timestamp,
                                                                             Timestamp,
                                                                             Location,
                                                                             Location,
                                                                             Location,
                                                                             Cost,
                                                                             InsertionData,
                                                                             InsertionData,
                                                                             InsertionData>({
    {5, 9, 5, 9, 3, 7, 11, 8, {0, 3}, {2, 7}, {3, 11}},  // s 3 [5] 7 11 [9] e
  }));

  GIVEN("tour with one activity and timing constraint") {
    auto fleet = createFleet();
    auto context =
      test_build_insertion_context{}
        .problem(createProblem(fleet))
        .progress(test_build_insertion_progress{}.owned())
        .solution(build_insertion_solution_context{}
                    .routes({test_build_insertion_route_context{}
                               .insert(test_build_activity{}.location(loc1).schedule({sh1, sh1}).shared(), 1)
                               .insert(test_build_activity{}.location(loc2).schedule({sh2, sh2}).shared(), 2)
                               .owned()})

                    .registry(std::make_shared<Registry>(*fleet))
                    .shared())
        .owned();
    auto rs = *context.solution->routes.begin();
    context.problem->constraint->accept(rs);

    WHEN("sequence is inserted with activities with relaxed tw") {
      THEN("returns insertion success with two activities and proper cost") {
        auto result =
          InsertionEvaluator{}.evaluate(as_job(test_build_sequence{}
                                                 .id("sequence")
                                                 .service(test_build_service{}.id("s1").location(s1).shared())
                                                 .service(test_build_service{}.id("s2").location(s2).shared())
                                                 .service(test_build_service{}.id("s3").location(s3).shared())
                                                 .shared()),
                                        context);
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).cost == cost);
        assertActivities(ranges::get<0>(result), {r1, r2, r3});
      }
    }
  }
}

SCENARIO("insertion evaluator can insert sequence in tour with two activities",
         "[algorithms][construction][insertion][sequence]") {
  auto [exst1, sched1, exst2, sched2, s1, s2, cost, r1, r2] =
    GENERATE(table<Location, Timestamp, Location, Timestamp, Location, Location, Cost, InsertionData, InsertionData>({
      {3, 3, 7, 7, 1, 9, 8, {0, 1}, {2, 9}},  // s  1 [3]  9 [7] e
      {7, 7, 3, 3, 9, 1, 8, {0, 9}, {3, 1}},  // s  9 [7] [3] 1  e
      {7, 7, 3, 3, 9, 5, 8, {0, 9}, {2, 5}},  // s  9 [7]  5 [3] e
    }));

  GIVEN("tour with one activity and timing constraint") {
    auto fleet = createFleet();
    auto context =
      test_build_insertion_context{}
        .problem(createProblem(fleet))
        .progress(test_build_insertion_progress{}.owned())
        .solution(build_insertion_solution_context{}
                    .routes({test_build_insertion_route_context{}
                               .insert(test_build_activity{}.location(exst1).schedule({sched1, sched1}).shared(), 1)
                               .insert(test_build_activity{}.location(exst2).schedule({sched2, sched2}).shared(), 2)
                               .owned()})

                    .registry(std::make_shared<Registry>(*fleet))
                    .shared())
        .owned();
    auto rs = *context.solution->routes.begin();
    context.problem->constraint->accept(rs);

    WHEN("sequence is inserted with activities with relaxed tw") {
      THEN("returns insertion success with two activities and proper cost") {
        auto result =
          InsertionEvaluator{}.evaluate(as_job(test_build_sequence{}
                                                 .id("sequence")
                                                 .service(test_build_service{}.id("s1").location(s1).shared())
                                                 .service(test_build_service{}.id("s2").location(s2).shared())
                                                 .shared()),
                                        context);
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).cost == cost);
        assertActivities(ranges::get<0>(result), {r1, r2});
      }
    }
  }
}

SCENARIO("insertion evaluator can handle sequence insertion in tour with violation",
         "[algorithms][construction][insertion][sequence]") {
  for (auto hasActivityInTour : std::vector<bool>{true, false}) {
    auto builder = test_build_insertion_context{};
    if (hasActivityInTour)
      builder.solution(build_insertion_solution_context{}
                         .routes({test_build_insertion_route_context{}
                                    .insert(test_build_activity{}.location(5).schedule({5, 5}).shared(), 1)
                                    .owned()})
                         .registry(std::make_shared<Registry>(*createFleet()))
                         .shared());

    GIVEN("failed route constraint") {
      auto constraint = std::make_shared<InsertionConstraint>();
      constraint->addHardRoute(
        std::make_shared<HardRouteWrapper>([](const auto&, const auto&) { return HardRouteConstraint::Result{42}; }));

      WHEN("sequence is evaluated") {
        auto result =
          InsertionEvaluator{}.evaluate(DefaultSequence, builder.problem(createProblem(constraint)).owned());

        THEN("returns insertion failure with proper code") {
          REQUIRE(result.index() == 1);
          REQUIRE(ranges::get<1>(result).constraint == 42);
        }
      }
    }

    GIVEN("failed activity constraint for any service") {
      auto constraint = std::make_shared<InsertionConstraint>();
      constraint->addHardActivity(std::make_shared<HardActivityWrapper>([](const auto&, const auto&) {
        return HardActivityConstraint::Result{{true, 42}};
      }));

      WHEN("sequence is evaluated") {
        auto result =
          InsertionEvaluator{}.evaluate(DefaultSequence, builder.problem(createProblem(constraint)).owned());

        THEN("returns insertion failure with proper code") {
          REQUIRE(result.index() == 1);
          REQUIRE(ranges::get<1>(result).constraint == 42);
        }
      }
    }

    GIVEN("failed activity constraint for second service only") {
      auto constraint = std::make_shared<InsertionConstraint>();
      constraint->addHardActivity(std::make_shared<HardActivityWrapper>([](const auto&, const auto& aCtx) {
        return aCtx.prev->detail.location == 0 ? HardActivityConstraint::Result{}
                                               : HardActivityConstraint::Result{{true, 42}};
      }));

      WHEN("sequence is evaluated") {
        auto result =
          InsertionEvaluator{}.evaluate(DefaultSequence, builder.problem(createProblem(constraint)).owned());

        THEN("returns insertion failure with proper code") {
          REQUIRE(result.index() == 1);
          REQUIRE(ranges::get<1>(result).constraint == 42);
        }
      }
    }
  }
}

// endregion
}
