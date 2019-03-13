#include "algorithms/construction/constraints/ConditionalJob.hpp"

#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/models/Helpers.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace ranges;

namespace vrp::test {

SCENARIO("job can be promoted from optional list", "[algorithms][construction][constraints]") {
  GIVEN("three jobs in optional list") {
    auto ctx = InsertionSolutionContext{
      {},
      {
        as_job(test_build_service{}.id("s0").shared()),
        as_job(test_build_service{}.id("s1").shared()),
        as_job(test_build_service{}.id("s2").shared()),
      },
      {},
      {},
      {},
    };

    WHEN("accepts solution change") {
      ConditionalJob{[](const auto&, const auto& job) { return get_job_id{}(job) == "s1"; }}.accept(ctx);

      THEN("one is promoted to required") {
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(ctx.required), Catch::Equals(std::vector<std::string>{"s1"}));
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(ctx.optional),
                   Catch::Equals(std::vector<std::string>{"s0", "s2"}));
      }
    }
  }
}

SCENARIO("job can be promoted from required list", "[algorithms][construction][constraints]") {
  GIVEN("three jobs in optional list") {
    auto ctx = InsertionSolutionContext{
      {
        as_job(test_build_service{}.id("s0").shared()),
        as_job(test_build_service{}.id("s1").shared()),
        as_job(test_build_service{}.id("s2").shared()),
      },
      {},
      {},
      {},
      {},
    };

    WHEN("accepts solution change") {
      ConditionalJob{[](const auto&, const auto& job) { return get_job_id{}(job) != "s1"; }}.accept(ctx);

      THEN("one is promoted to optional") {
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(ctx.required),
                   Catch::Equals(std::vector<std::string>{"s0", "s2"}));
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(ctx.optional), Catch::Equals(std::vector<std::string>{"s1"}));
      }
    }
  }
}

SCENARIO("jobs can be moved across required and optional lists", "[algorithms][construction][constraints]") {
  GIVEN("jobs in required and optional lists") {
    auto ctx = InsertionSolutionContext{
      {
        as_job(test_build_service{}.id("s0").shared()),
        as_job(test_build_service{}.id("s1").shared()),
        as_job(test_build_service{}.id("s2").shared()),
      },
      {
        as_job(test_build_service{}.id("s3").shared()),
        as_job(test_build_service{}.id("s4").shared()),
        as_job(test_build_service{}.id("s5").shared()),
      },
      {},
      {},
      {},
    };

    WHEN("accepts solution change") {
      ConditionalJob{[](const auto&, const auto& job) {
        auto id = get_job_id{}(job);
        return id != "s1" && id != "s3" && id != "s5";
      }}
        .accept(ctx);

      THEN("jobs are moved across lists according to predicate") {
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(ctx.required),
                   Catch::Equals(std::vector<std::string>{"s0", "s2", "s4"}));
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(ctx.optional),
                   Catch::Equals(std::vector<std::string>{"s1", "s3", "s5"}));
      }
    }
  }
}
}