#include "algorithms/refinement/extensions/CreateRefinementContext.hpp"

#include "streams/in/scientific/Solomon.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::algorithms::refinement;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::streams::in;
using namespace vrp::utils;
using namespace ranges;
using namespace Catch;

using Locks = std::vector<vrp::models::Lock>;

namespace {

Lock::Jobs
getJobs(const Problem& problem, bool isEven = true) {
  auto jobs = Lock::Jobs{};

  ranges::copy(view::zip(problem.jobs->all(), view::iota(1)) |
                 view::filter([&](const auto& pair) { return pair.second % 2 == (isEven ? 0 : 1); }) |
                 view::transform([](const auto& pair) { return pair.first; }),
               ranges::inserter(jobs, jobs.begin()));

  return jobs;
}
}

namespace vrp::test {

SCENARIO("create refinement context adds job locks to locked with sequence or strict order",
         "[algorithms][refinement][extensions]") {
  auto order = GENERATE(Lock::Order::Sequence, Lock::Order::Strict);

  GIVEN("Problem with jobs lock and one vehicle") {
    auto stream = create_sequential_problem_stream{}(1, 10);
    auto problem = read_solomon_type<cartesian_distance>{}(stream);
    problem->locks = std::make_shared<Locks>(Locks{
      Lock{[](const auto&) { return true; }, {Lock::Detail{order, Lock::Position::middle(), getJobs(*problem)}}}});

    WHEN("create refinement context") {
      auto ctx = create_refinement_context{}(problem);
      THEN("has expected locked jobs") {
        CHECK_THAT(get_job_ids_from_jobs{}(*ctx.locked), Catch::Equals(std::vector<std::string>{"c2", "c4"}));
      }

      THEN("has one route with all jobs") {
        REQUIRE(ctx.population->front().first->routes.size() == 1);
        CHECK_THAT(get_job_ids_from_all_routes{}(*ctx.population->front().first) | action::sort,
                   Catch::Equals(std::vector<std::string>{"c1", "c2", "c3", "c4", "c5"}));
      }
    }
  }
}

SCENARIO("create refinement context sticks jobs to actor with sequence or strict order",
         "[algorithms][refinement][extensions]") {
  auto order = GENERATE(Lock::Order::Sequence, Lock::Order::Strict);
  GIVEN("Problem with jobs lock and two vehicles") {
    auto stream = create_sequential_problem_stream{}(2, 10);
    auto problem = read_solomon_type<cartesian_distance>{}(stream);
    problem->locks =
      std::make_shared<Locks>(Locks{Lock{[](const auto& a) { return get_vehicle_id{}(*a.vehicle) == "v1"; },
                                         {Lock::Detail{order, Lock::Position::middle(), getJobs(*problem, true)}}},
                                    Lock{[](const auto& a) { return get_vehicle_id{}(*a.vehicle) == "v2"; },
                                         {Lock::Detail{order, Lock::Position::middle(), getJobs(*problem, false)}}}});

    WHEN("refinement context created") {
      auto ctx = create_refinement_context{}(problem);

      auto solution = ctx.population->front().first;
      auto route1 = find_route_by_vehicle_id{}(*solution, "v1");
      auto route2 = find_route_by_vehicle_id{}(*solution, "v2");

      THEN("initial solution has two routes") {
        REQUIRE(solution->routes.size() == 2);
        REQUIRE(route1.has_value());
        REQUIRE(route2.has_value());
      }

      THEN("initial solution has expected job order in routes") {
        CHECK_THAT(get_job_ids_from_all_routes{}(route1.value()), Equals(std::vector<std::string>{"c2", "c4"}));
        CHECK_THAT(get_job_ids_from_all_routes{}(route2.value()), Equals(std::vector<std::string>{"c1", "c3", "c5"}));
      }
    }
  }
}

SCENARIO("create refinement context can handle locked jobs as unassigned jobs",
         "[algorithms][refinement][extensions]") {
  auto order = GENERATE(Lock::Order::Any, Lock::Order::Sequence, Lock::Order::Strict);
  GIVEN("Problem with jobs lock and two vehicles") {
    auto stream = create_sequential_problem_stream{}(1, 10);
    auto problem = read_solomon_type<cartesian_distance>{}(stream);
    problem->locks = std::make_shared<Locks>(Locks{
      Lock{[](const auto& a) { return false; }, {Lock::Detail{order, Lock::Position::middle(), getJobs(*problem)}}}});

    WHEN("refinement context created") {
      auto ctx = create_refinement_context{}(problem);
      auto solution = ctx.population->front().first;

      THEN("route has expected locked jobs") {
        CHECK_THAT(get_job_ids_from_all_routes{}(*solution) | action::sort,
                   Catch::Equals(std::vector<std::string>{"c1", "c3", "c5"}));
      }

      THEN("has unassigned jobs") { REQUIRE(solution->unassigned.size() == 2); }
    }
  }
}
}
