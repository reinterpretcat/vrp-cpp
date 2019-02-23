#include "algorithms/construction/extensions/Sorters.hpp"

#include "models/extensions/problem/Helpers.hpp"
#include "test_utils/algorithms/construction/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace ranges;
using namespace Catch;

using Demand = typename VehicleActivitySize<int>::Demand;

namespace {

std::vector<std::string>
extractJobIds(const InsertionContext& ctx) {
  return ctx.jobs | view::transform([](const auto& job) { return vrp::models::problem::get_job_id{}(job); }) |
    to_vector;
}
}

namespace vrp::test {

SCENARIO("sized jobs sorter can sort by job demand", "[algorithms][construction][sorters]") {
  GIVEN("unsorted two services and one sequence") {
    auto ctx = test_build_insertion_context{}
                 .jobs({as_job(test_build_service{}.id("srv1").demand<int>(Demand{{0, 0}, {3, 0}}).shared()),
                        as_job(test_build_service{}.id("srv2").demand<int>(Demand{{0, 0}, {10, 0}}).shared()),
                        as_job(test_build_sequence{}
                                 .id("seq1")
                                 .service(test_build_service{}.demand<int>(Demand{{0, 0}, {5, 0}}).shared())
                                 .shared())})
                 .registry({})
                 .problem({})
                 .owned();

    WHEN("sort by size") {
      sized_jobs_sorter<int>{}(ctx);

      THEN("should have jobs sorted in desc order") {
        CHECK_THAT(extractJobIds(ctx), Equals(std::vector<std::string>{"srv2", "seq1", "srv1"}));
      }
    }
  }
}
}