#include "algorithms/construction/extensions/Sorters.hpp"

#include "models/extensions/problem/Helpers.hpp"
#include "test_utils/algorithms/construction/Factories.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace ranges;
using namespace Catch;

using Demand = typename VehicleActivitySize<int>::Demand;

namespace {

std::vector<std::string>
extractJobIds(const InsertionContext& ctx) {
  return ctx.solution->required |
    view::transform([](const auto& job) { return vrp::models::problem::get_job_id{}(job); }) | to_vector;
}
}

namespace vrp::test {

SCENARIO("sized jobs sorter can sort by job demand", "[algorithms][construction][sorters]") {
  auto [isDesc, ids] = GENERATE(table<bool, std::vector<std::string>>({
    {true, {"srv2", "seq1", "srv1"}},
    {false, {"srv1", "seq1", "srv2"}},
  }));
  GIVEN("unsorted two services and one sequence") {
    auto ctx = test_build_insertion_context{}
                 .jobs({as_job(test_build_service{}.id("srv1").demand<int>(Demand{{0, 0}, {3, 0}}).shared()),
                        as_job(test_build_service{}.id("srv2").demand<int>(Demand{{0, 0}, {10, 0}}).shared()),
                        as_job(test_build_sequence{}
                                 .id("seq1")
                                 .service(test_build_service{}.demand<int>(Demand{{0, 0}, {5, 0}}).shared())
                                 .shared())})
                 .problem({})
                 .owned();

    WHEN("sort by size") {
      sized_jobs_sorter<int>{isDesc}(ctx);

      THEN("should have jobs sorted in proper order") { CHECK_THAT(extractJobIds(ctx), Equals(ids)); }
    }
  }
}

SCENARIO("ranked jobs sorter can sort by distance rank", "[algorithms][construction][sorters]") {
  auto [isDesc, ids] = GENERATE(table<bool, std::vector<std::string>>({
    {false, {"seq0", "srv0", "srv2", "srv1"}},
    {true, {"srv1", "srv2", "srv0", "seq0"}},
  }));

  auto fleet = std::make_shared<Fleet>();
  (*fleet)
    .add(test_build_driver{}.owned())
    .add(test_build_vehicle{}.id("v1").profile("p1").details({{0, 0, {0, 100}}}).owned());
  auto species = std::vector<models::problem::Job>{
    as_job(test_build_service{}.location(5).id("srv0").shared()),
    as_job(test_build_service{}.location(20).id("srv1").shared()),
    as_job(test_build_service{}.location(15).id("srv2").shared()),
    as_job(test_build_sequence{}.id("seq0").service(test_build_service{}.location(0).shared()).shared())};
  auto jobs = std::make_shared<Jobs>(Jobs{TestTransportCosts{}, *fleet, ranges::view::all(species)});

  GIVEN("unsorted tree services and one sequence") {
    auto ctx = test_build_insertion_context{}
                 .jobs(std::move(species))
                 .problem(std::make_shared<Problem>(Problem{fleet, jobs, {}, {}, {}, {}, {}}))
                 .owned();

    WHEN("sort by rank") {
      ranked_jobs_sorter{isDesc}(ctx);

      THEN("should have jobs sorted in proper order") { CHECK_THAT(extractJobIds(ctx), Equals(ids)); }
    }
  }
}
}
