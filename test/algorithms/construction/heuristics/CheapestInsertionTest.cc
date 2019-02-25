#include "algorithms/construction/heuristics/CheapestInsertion.hpp"

#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "streams/in/LiLim.hpp"
#include "streams/in/Solomon.hpp"
#include "test_utils/algorithms/construction/Factories.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/streams/LiLimStreams.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::streams::in;
using namespace Catch::Matchers;
using namespace ranges;

namespace {

template<typename ReaderType = read_solomon_type<cartesian_distance>>
std::tuple<InsertionEvaluator, InsertionContext>
createInsertion(std::stringstream stream) {
  auto problem = ReaderType{}.operator()(stream);
  auto ctx = vrp::test::test_build_insertion_context{}
               .progress(vrp::test::test_build_insertion_progress{}.total(problem->jobs->size()).owned())
               .jobs(problem->jobs->all())
               .registry(std::make_shared<Registry>(*problem->fleet))
               .problem(problem)
               .owned();

  return {{}, ctx};
}

template<typename ProblemStream, typename ReaderType = read_solomon_type<cartesian_distance>>
std::tuple<InsertionEvaluator, InsertionContext>
createInsertion(int vehicles, int capacities) {
  return createInsertion<ReaderType>(ProblemStream{}(vehicles, capacities));
}

template<typename ProblemStream, typename ReaderType = read_solomon_type<cartesian_distance>>
std::tuple<InsertionEvaluator, InsertionContext>
createInsertion() {
  return createInsertion<ReaderType>(ProblemStream{}());
}
}

namespace vrp::test {

// region Service

SCENARIO("cheapest insertion inserts service", "[algorithms][construction][insertion][service]") {
  using EndLoc = std::optional<Location>;

  auto [s1, v1, v2, used] = GENERATE(table<Location, EndLoc, EndLoc, std::string>({
    {3, {}, {}, "v1"},
    {21, {}, {}, "v2"},
  }));

  GIVEN("one service job and two vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, v1, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{20, v2, {0, 100}}}).owned());
    auto constraint = std::make_shared<InsertionConstraint>();
    auto problem = std::make_shared<models::Problem>(models::Problem{
      {}, {}, constraint, {}, std::make_shared<ActivityCosts>(), std::make_shared<TestTransportCosts>()});

    constraint->add<ActorActivityTiming>(
      std::make_shared<ActorActivityTiming>(fleet, problem->transport, problem->activity));

    auto insertion = CheapestInsertion{InsertionEvaluator{}};

    WHEN("analyzes insertion context") {
      auto result = insertion(test_build_insertion_context{}
                                .registry(std::make_shared<Registry>(*fleet))
                                .problem(problem)
                                .jobs({as_job(test_build_service{}.location(s1).shared())})
                                .owned());

      THEN("returns new context with job inserted") {
        REQUIRE(result.unassigned.empty());
        REQUIRE(result.routes.size() == 1);
        REQUIRE(get_vehicle_id{}(*result.routes.begin()->route->actor->vehicle) == used);
        REQUIRE(result.routes.begin()->route->tour.get(1)->detail.location == s1);
      }
    }
  }
}

SCENARIO("cheapest insertion handles artificial problems with demand",
         "[algorithms][construction][insertion][service]") {
  //  auto [vehicles, capacity, unassigned, routes] =
  //    GENERATE(table<int, int, int, int>({{1, 10, 0, 1}, {2, 4, 0, 2}, {1, 4, 1, 1}, {1, 3, 2, 1}}));
  // TODO what is wrong with generator here?
  for (auto [vehicles, capacity, unassigned, routes] :
       std::vector<std::tuple<int, int, int, int>>{{1, 10, 0, 1}, {2, 4, 0, 2}, {1, 4, 1, 1}, {1, 3, 2, 1}}) {
    GIVEN("sequential coordinates problem") {
      auto [evaluator, ctx] = createInsertion<create_sequential_problem_stream>(vehicles, capacity);

      WHEN("calculates solution") {
        auto solution = CheapestInsertion{evaluator}.operator()(ctx);
        THEN("all jobs processed") {
          REQUIRE(solution.jobs.empty());
          REQUIRE(solution.unassigned.size() == unassigned);
          REQUIRE(solution.routes.size() == routes);
        }
      }
    }
  }
}

SCENARIO("cheapest insertion handles artificial problems with times",
         "[algorithms][construction][insertion][service]") {
  GIVEN("time problem") {
    auto [evaluator, ctx] = createInsertion<create_time_problem_stream>(1, 10);

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("all jobs processed") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        REQUIRE(get_job_ids_from_all_routes{}.operator()(solution).front() == "c5");
      }
    }
  }
}

SCENARIO("cheapest insertion handles artificial problems with waiting",
         "[algorithms][construction][insertion][service]") {
  GIVEN("time problem") {
    struct create_waiting_problem_stream {
      std::stringstream operator()() {
        return SolomonBuilder()
          .setVehicle(1, 10)
          .addCustomer({0, 0, 0, 0, 0, 1000, 0})
          .addCustomer({1, 1, 0, 1, 20, 40, 10})
          .addCustomer({2, 2, 0, 1, 50, 100, 10})
          .build();
      }
    };
    auto [evaluator, ctx] = createInsertion<create_waiting_problem_stream>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("all jobs processed") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(solution), Equals(std::vector<std::string>{"c1", "c2"}));
      }
    }
  }
}

SCENARIO("cheapest insertion handles two customers with one route", "[algorithms][construction][insertion][service]") {
  GIVEN("two customers with strict tw") {
    struct create_timing_problem_stream {
      std::stringstream operator()() {
        return SolomonBuilder()
          .setVehicle(25, 200)
          .addCustomer({0, 40, 50, 0, 0, 1236, 0})
          .addCustomer({1, 45, 68, 10, 912, 967, 90})
          .addCustomer({2, 45, 70, 30, 825, 870, 90})
          .build();
      }
    };
    auto [evaluator, ctx] = createInsertion<create_timing_problem_stream>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("has solution with one route") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(solution), Equals(std::vector<std::string>{"c2", "c1"}));
      }
    }
  }
}

SCENARIO("cheapest insertion handles cannot handle two customers with one route",
         "[algorithms][construction][insertion][service]") {
  GIVEN("two customers with strict tw") {
    struct create_timing_problem_stream {
      std::stringstream operator()() {
        return SolomonBuilder()
          .setVehicle(25, 200)
          .addCustomer({0, 40, 50, 0, 0, 1236, 0})
          .addCustomer({5, 42, 65, 10, 15, 67, 90})
          .addCustomer({13, 22, 75, 30, 30, 92, 90})
          .build();
      }
    };
    auto [evaluator, ctx] = createInsertion<create_timing_problem_stream>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("has solution with two routes") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.routes.size() == 2);
      }
    }
  }
}

SCENARIO("cheapest insertion handles solomon set problems", "[algorithms][construction][insertion][service]") {
  GIVEN("c101_25 problem") {
    auto [evaluator, ctx] = createInsertion<create_c101_25_problem_stream>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);
      auto ids = get_job_ids_from_all_routes{}.operator()(solution);

      THEN("has expected solution") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(!solution.routes.empty());
        REQUIRE(solution.routes.size() <= 5);
        REQUIRE(ranges::accumulate(ids, 0, [](const auto acc, const auto next) { return acc + 1; }) == 25);
      }
    }
  }
}

// endregion

// region Sequence

SCENARIO("cheapest insertion handles two sequence insertion", "[algorithms][construction][insertion][sequence]") {
  GIVEN("simple problem with two sequences") {
    auto [evaluator, ctx] = createInsertion<create_two_sequences_stream, read_li_lim_type<cartesian_distance>>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("has expected solution") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        CHECK_THAT(get_service_ids_from_all_routes{}.operator()(solution),
                   Equals(std::vector<std::string>{"c3", "c1", "c2", "c4"}));
      }
    }
  }
}

SCENARIO("cheapest insertion handles five sequence insertion", "[algorithms][construction][insertion][sequence]") {
  GIVEN("simple problem with five sequences") {
    struct create_five_sequences_stream {
      std::stringstream operator()(int vehicles = 25, int capacity = 50) {
        return LiLimBuilder()
          .setVehicle(vehicles, capacity)
          .addCustomer({0, 40, 50, 0, 0, 1236, 0, 0, 0})
          .addCustomer({1, 42, 66, 10, 65, 146, 90, 0, 6})
          .addCustomer({2, 42, 65, 10, 15, 67, 90, 0, 7})
          .addCustomer({3, 40, 69, 20, 621, 702, 90, 0, 8})
          .addCustomer({4, 38, 68, 20, 255, 324, 90, 0, 9})
          .addCustomer({5, 38, 70, 10, 534, 605, 90, 0, 10})
          .addCustomer({6, 45, 65, -10, 997, 1068, 90, 1, 0})
          .addCustomer({7, 40, 66, -10, 170, 225, 90, 2, 0})
          .addCustomer({8, 45, 70, -20, 825, 870, 90, 3, 0})
          .addCustomer({9, 35, 66, -20, 357, 410, 90, 4, 0})
          .addCustomer({10, 42, 68, -10, 727, 782, 90, 5, 0})
          .build();
      }
    };

    auto [evaluator, ctx] = createInsertion<create_five_sequences_stream, read_li_lim_type<cartesian_distance>>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("has expected solution") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        CHECK_THAT(get_service_ids_from_all_routes{}.operator()(solution),
                   Equals(std::vector<std::string>{"c2", "c1", "c7", "c4", "c9", "c5", "c3", "c10", "c8", "c6"}));
      }
    }
  }
}

SCENARIO("cheapest insertion analyzes all insertion places", "[algorithms][construction][insertion][sequence]") {
  GIVEN("reference problem with two sequences") {
    struct create_reference_problem_stream {
      std::stringstream operator()(int vehicles = 250, int capacity = 200) {
        return LiLimBuilder()
          .setVehicle(vehicles, capacity)
          .addCustomer({0, 250, 250, 0, 0, 1821, 0, 0, 0})
          .addCustomer({65, 200, 261, 10, 166, 196, 10, 0, 610})
          .addCustomer({126, 201, 270, 10, 52, 82, 10, 0, 374})
          .addCustomer({349, 200, 270, -30, 59, 89, 10, 838, 0})
          .addCustomer({374, 200, 265, -10, 152, 182, 10, 126, 0})
          .addCustomer({464, 199, 268, 20, 71, 101, 10, 0, 967})
          .addCustomer({610, 206, 261, -10, 182, 212, 10, 65, 0})
          .addCustomer({838, 204, 269, 30, 49, 79, 10, 0, 349})
          .addCustomer({868, 198, 264, -20, 140, 170, 10, 976, 0})
          .addCustomer({967, 197, 270, -20, 96, 126, 10, 464, 0})
          .addCustomer({976, 198, 271, 20, 84, 114, 10, 0, 868})
          .build();
      }
    };

    auto [evaluator, ctx] = createInsertion<create_reference_problem_stream, read_li_lim_type<cartesian_distance>>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("has expected solution") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(solution),
                   Equals(std::vector<std::string>{
                     "seq3", "seq1", "seq3", "seq2", "seq4", "seq2", "seq1", "seq4", "seq0", "seq0"}));
      }
    }
  }
}

SCENARIO("cheapest insertion handles edge case with first best but worst in total",
         "[algorithms][construction][insertion][sequence]") {
  GIVEN("reference problem with five sequences") {
    /// Reproduces the problem when later best first service insertion leads to worse total insertion, e.g.:
    /// seq3 seq1 seq3 seq4 seq2 seq2 seq1 seq4 seq0 seq0 (worse)
    /// seq3 seq1 seq3 seq2 seq4 seq2 seq1 seq4 seq0 seq0 (best)
    struct create_reference_problem_stream {
      std::stringstream operator()(int vehicles = 250, int capacity = 200) {
        return LiLimBuilder()
          .setVehicle(vehicles, capacity)
          .addCustomer({0, 250, 250, 0, 0, 1821, 0, 0, 0})
          .addCustomer({65, 200, 261, 10, 166, 196, 10, 0, 610})
          .addCustomer({126, 201, 270, 10, 52, 82, 10, 0, 374})
          .addCustomer({349, 200, 270, -30, 59, 89, 10, 838, 0})
          .addCustomer({374, 200, 265, -10, 152, 182, 10, 126, 0})
          .addCustomer({464, 199, 268, 20, 71, 101, 10, 0, 967})
          .addCustomer({610, 206, 261, -10, 182, 212, 10, 65, 0})
          .addCustomer({838, 204, 269, 30, 49, 79, 10, 0, 349})
          .addCustomer({868, 198, 264, -20, 140, 170, 10, 976, 0})
          .addCustomer({967, 197, 270, -20, 96, 126, 10, 464, 0})
          .addCustomer({976, 198, 271, 20, 84, 114, 10, 0, 868})
          .build();
      }
    };

    auto [evaluator, ctx] = createInsertion<create_reference_problem_stream, read_li_lim_type<cartesian_distance>>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("has expected solution") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(solution),
                   Equals(std::vector<std::string>{
                     "seq3", "seq1", "seq3", "seq2", "seq4", "seq2", "seq1", "seq4", "seq0", "seq0"}));
      }
    }
  }
}

SCENARIO("cheapest insertion handles edge case with failed insertion at the beginning",
         "[algorithms][construction][insertion][sequence]") {
  GIVEN("reference problem with three sequences") {
    struct create_reference_problem_stream {
      std::stringstream operator()(int vehicles = 250, int capacity = 200) {
        return LiLimBuilder()
          .setVehicle(vehicles, capacity)
          .addCustomer({0, 250, 250, 0, 0, 1821, 0, 0, 0})
          .addCustomer({63, 388, 331, -30, 160, 190, 10, 792, 0})
          .addCustomer({315, 394, 333, 20, 194, 224, 10, 0, 722})
          .addCustomer({518, 388, 334, 20, 161, 191, 10, 0, 685})
          .addCustomer({685, 389, 334, -20, 169, 199, 10, 518, 0})
          .addCustomer({722, 371, 315, -20, 1166, 1196, 10, 315, 0})
          .addCustomer({792, 321, 280, 30, 77, 107, 10, 0, 63})
          .build();
      }
    };

    auto [evaluator, ctx] = createInsertion<create_reference_problem_stream, read_li_lim_type<cartesian_distance>>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("has expected solution") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(solution),
                   Equals(std::vector<std::string>{"seq2", "seq2", "seq1", "seq1", "seq0", "seq0"}));
      }
    }
  }
}

/// endregion

// region Unassigned

SCENARIO("cheapest insertion handles unassigned job with capacity reason",
         "[algorithms][construction][insertion][unassigned]") {
  GIVEN("two customers with strict tw") {
    struct create_timing_problem_stream {
      std::stringstream operator()() {
        return SolomonBuilder()
          .setVehicle(1, 10)
          .addCustomer({0, 0, 0, 0, 0, 1236, 0})
          .addCustomer({1, 1, 0, 5, 0, 100, 0})
          .addCustomer({2, 2, 0, 1, 0, 100, 0})
          .addCustomer({3, 3, 0, 5, 0, 100, 0})
          .addCustomer({4, 4, 0, 1, 0, 100, 0})
          .build();
      }
    };
    auto [evaluator, ctx] = createInsertion<create_timing_problem_stream>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion{evaluator}.operator()(ctx);

      THEN("has solution with one unassigned") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.size() == 1);
        REQUIRE(solution.unassigned.begin()->second == 2);
        REQUIRE(solution.routes.size() == 1);
      }
    }
  }
}

// endregion

SCENARIO("Can solve simple open VRP problem", "[scenarios][openvrp]") {
  auto [jobs, size] = GENERATE(std::make_tuple(
                                 std::vector<Job>{
                                   as_job(test_build_service{}.location(5).shared()),
                                   as_job(test_build_service{}.location(10).shared()),
                                 },
                                 3),
                               std::make_tuple(std::vector<Job>{as_job(test_build_service{}.location(5).shared())}, 2));

  GIVEN("An open VRP problem with one or more jobs") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet).add(test_build_driver{}.owned()).add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned());

    auto activity = std::make_shared<ActivityCosts>();
    auto transport = std::make_shared<TestTransportCosts>();

    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>());
    constraint->add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, transport, activity));

    auto problem = std::make_shared<models::Problem>(models::Problem{{}, {}, constraint, {}, activity, transport});

    WHEN("run solver") {
      auto solution = CheapestInsertion{InsertionEvaluator{}}.operator()(test_build_insertion_context{}
                                                                           .registry(std::make_shared<Registry>(*fleet))
                                                                           .problem(problem)
                                                                           .jobs(std::move(jobs))
                                                                           .owned());

      THEN("has proper tour end") {
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        REQUIRE(ranges::size(solution.routes.begin()->route->tour.activities()) == size);
        REQUIRE(solution.routes.begin()->route->tour.end()->detail.location != 0);
      }
    }
  }
}
}
