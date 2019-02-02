#include "streams/in/LiLim.hpp"

#include "models/extensions/problem/Helpers.hpp"
#include "test_utils/streams/LiLimStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::common;
using namespace vrp::models::problem;
using namespace vrp::streams::in;
using namespace vrp::test;
using namespace Catch::Matchers;
using namespace ranges;

namespace vrp::test {

SCENARIO("lilim files can be read from input stream", "[streams][in]") {
  GIVEN("lc101 with 100 customers") {
    auto stream = create_lc101_problem_stream{}();

    WHEN("cartesian distances are used") {
      auto problem = read_li_lim_type<>{}(stream);

      THEN("has 53 jobs") {
        auto ids = problem->jobs->all() | to_vector;

        REQUIRE(ids.size() == 53);
      }
    }
  }
}
}
