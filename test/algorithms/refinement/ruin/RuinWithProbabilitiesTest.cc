#include "algorithms/refinement/ruin/RuinWithProbabilities.hpp"

#include "test_utils/fakes/FakeDistribution.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::algorithms::refinement;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace vrp::utils;

namespace {

struct First {
  void operator()(const RefinementContext& rCtx, const Solution& sln, InsertionContext& iCtx) {
    iCtx.unassigned.insert({as_job(vrp::test::test_build_service{}.shared()), 1});
  }
};

struct Second {
  void operator()(const RefinementContext& rCtx, const Solution& sln, InsertionContext& iCtx) {
    iCtx.unassigned.insert({as_job(vrp::test::test_build_service{}.shared()), 2});
  }
};

struct Third {
  void operator()(const RefinementContext& rCtx, const Solution& sln, InsertionContext& iCtx) {
    iCtx.unassigned.insert({as_job(vrp::test::test_build_service{}.shared()), 3});
  }
};
}

namespace vrp::test {

SCENARIO("ruin with probabilities can ruin the same context", "[algorithms][refinement][ruin][service]") {
  auto [doubles, codes] = GENERATE(table<std::vector<double>, std::vector<int>>({
    {{0.4, 0.6, 0.7}, {1, 2, 3}},
    {{0.6, 0.8, 0.7}, {2, 3}},
    {{0.6, 1, 0.9}, {2}},
  }));

  GIVEN("three refinement context") {
    auto random = std::make_shared<Random>(FakeDistribution<int>{{}, 0}, FakeDistribution<double>{doubles, 0});
    auto rCtx = RefinementContext{{}, random, {}, {}, 0};
    auto iCtx = InsertionContext{{}, {}, {}, {}, {}, {}, {}};

    WHEN("ruin with different probabilities") {
      ruin_with_probabilities<std::tuple<First, Probability<5, 10>>,
                              std::tuple<Second, Probability<10, 10>>,
                              std::tuple<Third, Probability<8, 10>>>{}
        .
        operator()(rCtx, {}, iCtx);

      THEN("should run expected ruin strategies") {
        REQUIRE(iCtx.unassigned.size() == codes.size());
        CHECK_THAT(iCtx.unassigned | ranges::view::transform([](const auto& pair) { return pair.second; }) |
                     ranges::to_vector | ranges::action::sort,
                   Catch::Equals(codes));
      }
    }
  }
}
}