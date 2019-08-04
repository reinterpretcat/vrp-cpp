#include "utils/Permutations.hpp"

#include <algorithm>
#include <catch/catch.hpp>

using namespace vrp::utils;

namespace vrp::test {

SCENARIO("can generate permutations of given range", "[utils][permutations]") {
  auto [rangeSize, expectedSize] = GENERATE(table<int, int>({{3, 6}, {4, 24}}));

  GIVEN("permutation generator range within size") {
    auto range = permutation_range{0, rangeSize - 1};

    WHEN("generate permutation") {
      auto result = range | ranges::view::transform([](const auto& permutation) {
                      return std::vector<int>(permutation.begin(), permutation.end());
                    }) |
        ranges::to_vector;

      THEN("returns all possible permutations") { REQUIRE(result.size() == expectedSize); }
    }
  }
}

SCENARIO("can generate permutations of two sets", "[utils][permutations]") {
  auto size = 7;

  GIVEN("permutation set generator within range and limit") {
    std::mt19937 engine;

    WHEN("generate permutations") {
      auto permutations = generate_set_permutations{}(2, 5, 7, engine);

      THEN("returns specific amount of permutations") { REQUIRE(ranges::distance(permutations) == size); }
    }
  }
}
}
