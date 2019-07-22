#include "utils/Permutations.hpp"

#include <algorithm>
#include <catch/catch.hpp>

using namespace vrp::utils;

namespace vrp::test {

SCENARIO("can generate permutations of given range", "[utils][permutations]") {
  auto [rangeSize, expectedSize] = GENERATE(table<int, int>({ {3, 6}, {4, 24} }));

  GIVEN("permutation generator within size") {
    auto generator = generate_permutations{rangeSize};

    WHEN("generate permutation") {
      auto result = generator() | ranges::view::transform([](const auto& permutation) {
                      return std::vector<int>(permutation.begin(), permutation.end());
                    }) |
        ranges::to_vector;

      THEN("returns all possible permutations") { REQUIRE(result.size() == expectedSize); }
    }
  }
}

}
