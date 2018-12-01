#include "streams/in/Solomon.hpp"

#include "test_utils/streams/SolomonBuilder.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::common;
using namespace vrp::streams::in;
using namespace vrp::test;
using namespace Catch::Matchers;
using namespace ranges;

namespace {
struct WithSimplifiedCoordinates {
  std::stringstream operator()() {
    return SolomonBuilder()
      .setTitle("Customers with simplified coordinates")
      .setVehicle(2, 10)
      .addCustomer({0, 0, 0, 0, 0, 1000, 1})
      .addCustomer({1, 1, 0, 1, 5, 1000, 5})
      .addCustomer({2, 3, 0, 2, 0, 1002, 11})
      .addCustomer({3, 7, 0, 1, 0, 1000, 12})
      .build();
  }
};

}  // namespace

namespace vrp::test {

SCENARIO("solomon files can be read from input stream", "[streams][in]") {
  GIVEN("simple problem") {
    auto stream = WithSimplifiedCoordinates()();

    WHEN("cartesian distances are used") {
      auto result = read_solomon_type<scaled_cartesian_distance>{}(stream);

      THEN("jobs have proper ids") {
        auto ids = std::get<0>(result).jobs | view::transform([](const auto& job) { return ranges::get<0>(job)->id; });

        CHECK_THAT(ids, Equals(std::vector<std::string>{"c0", "c1", "c2", "c3"}));
      }

      THEN("jobs have proper demand") {
        auto demands = std::get<0>(result).jobs | view::transform([](const auto& job) {
                         return std::any_cast<int>(ranges::get<0>(job)->dimens.find("demand")->second);
                       });

        CHECK_THAT(demands, Equals(std::vector<int>{0, 1, 2, 1}));
      }

      THEN("jobs have proper service time") {
        auto durations = std::get<0>(result).jobs |
          view::transform([](const auto& job) { return ranges::get<0>(job)->details[0].duration; });

        CHECK_THAT(durations, Equals(std::vector<int>{1, 5, 11, 12}));
      }

      THEN("vehicles have proper ids") {
        std::vector<std::string> ids =
          std::get<0>(result).fleet->vehicles() | view::transform([](const auto& v) { return v->id; });

        CHECK_THAT(ids, Contains(std::vector<std::string>{"v1", "v2"}));
      }

      THEN("vehicles have proper capacity") {
        std::vector<int> capacities = std::get<0>(result).fleet->vehicles() |
          view::transform([](const auto& v) { return std::any_cast<int>(v->dimens.find("capacity")->second); });

        CHECK_THAT(capacities, Equals(std::vector<int>{10, 10}));
      }

      THEN("transport costs have expected matrix") {
        CHECK_THAT(std::get<2>(result)->matrix() | view::transform([](const auto& d) { return d / 1000; }),
                   Equals(std::vector<Distance>{0, 1, 3, 7, 1, 0, 2, 6, 3, 2, 0, 4, 7, 6, 4, 0}));
      }
    }
  }
}
}
