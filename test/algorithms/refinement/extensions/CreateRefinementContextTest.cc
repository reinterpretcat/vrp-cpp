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

namespace {

JobsLock::Jobs
getJobs(const Problem& problem) {
  auto jobs = JobsLock::Jobs{};

  ranges::copy(view::zip(problem.jobs->all(), view::iota(1)) |
                 view::filter([](const auto& pair) { return pair.second % 2 == 0; }) |
                 view::transform([](const auto& pair) { return pair.first; }),
               ranges::inserter(jobs, jobs.begin()));

  return jobs;
}
}

namespace vrp::test {

SCENARIO("create refinement context adds sequence or strict job locks to locked",
         "[algorithms][refinement][extensions]") {
  auto order = GENERATE(JobsLock::Order::Sequence, JobsLock::Order::Strict);

  GIVEN("Problem with jobs lock") {
    auto stream = create_sequential_problem_stream{}(1, 1);
    auto problem = read_solomon_type<cartesian_distance>{}(stream);
    auto locks = std::make_shared<std::vector<models::JobsLock>>();
    locks->push_back(JobsLock{[](const auto&) { return true; }, {JobsLock::Detail{order, getJobs(*problem)}}});
    problem->locks = locks;

    WHEN("create refinement context") {
      auto ctx = create_refinement_context{}(problem);
      THEN("has expected locked jobs") {
        CHECK_THAT(get_job_ids_from_jobs{}(*ctx.locked), Catch::Equals(std::vector<std::string>{"c2", "c4"}));
      }
    }
  }
}

// SCENARIO("create refinement context handles any lock", "[algorithms][refinement][extensions]") {
//
//}
//
// SCENARIO("create refinement context adds lock to unassigned jobs", "[algorithms][refinement][extensions]") {
//
//}
}
