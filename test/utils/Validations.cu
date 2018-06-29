#include "test_utils/PopulationFactory.hpp"
#include "test_utils/SolomonBuilder.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <catch/catch.hpp>

using namespace vrp::models;
using namespace vrp::utils;
using namespace vrp::test;

namespace {

struct WithSequentialCustomers {
  std::stringstream operator()() {
    return SolomonBuilder()
      .setTitle("Sequential customers")
      .setVehicle(1, 10)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 1, 0, 1, 0, 1000, 10})
      .addCustomer({2, 2, 0, 1, 0, 1000, 10})
      .addCustomer({3, 3, 0, 1, 0, 1000, 10})
      .addCustomer({4, 4, 0, 1, 0, 1000, 10})
      .addCustomer({5, 5, 0, 1, 0, 1000, 10})
      .build();
  }
};

}  // namespace

SCENARIO("Can check valid solution", "[utils][validation][solution_checker]") {
  // TODO
}