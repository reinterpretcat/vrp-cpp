#include "streams/in/json/detail/HereSequencePermutation.hpp"

#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>
#include <range/v3/all.hpp>

using namespace vrp::test;
using namespace Catch::Matchers;

namespace {
auto sequence = test_build_sequence{}
                  .diIndex(3)
                  .id("sequence")
                  .service(test_build_service{}.id("p1").location(1).shared())
                  .service(test_build_service{}.id("p2").location(2).shared())
                  .service(test_build_service{}.id("p3").location(3).shared())
                  .service(test_build_service{}.id("d1").location(4).shared())
                  .service(test_build_service{}.id("d2").location(5).shared())
                  .shared();
}

namespace vrp::test {

SCENARIO("here sequence permutation can generate all permutations") {
  GIVEN("sequence with three pickups, two deliveries, and permutation function") {
    auto permFunc = streams::in::detail::here::create_permutation_function{}(100);

    WHEN("permutations are generated") {
      auto permutations = permFunc->operator()(*sequence) | ranges::to<std::vector>;

      THEN("creates expected ranges") {
        CHECK_THAT(permutations,
                   Equals(std::vector<std::vector<int>>{{0, 1, 2, 3, 4},
                                                        {0, 1, 2, 4, 3},
                                                        {0, 2, 1, 3, 4},
                                                        {0, 2, 1, 4, 3},
                                                        {1, 0, 2, 3, 4},
                                                        {1, 0, 2, 4, 3},
                                                        {1, 2, 0, 3, 4},
                                                        {1, 2, 0, 4, 3},
                                                        {2, 0, 1, 3, 4},
                                                        {2, 0, 1, 4, 3},
                                                        {2, 1, 0, 3, 4},
                                                        {2, 1, 0, 4, 3}}));
      }
    }
  }
}

SCENARIO("here sequence permutation can limit amount of permutations") {
  GIVEN("sequence with three pickups, two deliveries, and permutation function") {
    auto permFunc = streams::in::detail::here::create_permutation_function{}(3);

    WHEN("permutations are generated") {
      auto permutations = permFunc->operator()(*sequence) | ranges::to_vector;

      THEN("creates permutation collection of expected size") { REQUIRE(permutations.size() == 3); }
    }
  }
}
}