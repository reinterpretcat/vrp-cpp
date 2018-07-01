#ifndef VRP_TESTUTILS_PROBLEMSTREAMS_HPP
#define VRP_TESTUTILS_PROBLEMSTREAMS_HPP

#include "test_utils/SolomonBuilder.hpp"
#include "utils/memory/DevicePool.hpp"

namespace vrp {
namespace test {

struct create_sequential_problem_stream {
  std::stringstream operator()(int capacity = 10) {
    return SolomonBuilder()
      .setTitle("Sequential customers")
      .setVehicle(1, capacity)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 1, 0, 1, 0, 1000, 10})
      .addCustomer({2, 2, 0, 1, 0, 1000, 10})
      .addCustomer({3, 3, 0, 1, 0, 1000, 10})
      .addCustomer({4, 4, 0, 1, 0, 1000, 10})
      .addCustomer({5, 5, 0, 1, 0, 1000, 10})
      .build();
  }
};

struct create_shuffled_coordinates {
  std::stringstream operator()(int capacity = 10) {
    return SolomonBuilder()
      .setTitle("Customers with shuffled coordinates")
      .setVehicle(1, capacity)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 2, 0, 1, 0, 1000, 10})
      .addCustomer({2, 4, 0, 1, 0, 1000, 10})
      .addCustomer({3, 1, 0, 1, 0, 1000, 10})
      .addCustomer({4, 5, 0, 1, 0, 1000, 10})
      .addCustomer({5, 3, 0, 1, 0, 1000, 10})
      .build();
  }
};

struct create_exceeded_capacity_variant_1_problem_stream {
  std::stringstream operator()() {
    return SolomonBuilder()
      .setTitle("Exceeded capacity and three vehicles")
      .setVehicle(3, 10)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 1, 0, 8, 0, 1000, 0})
      .addCustomer({2, 2, 0, 8, 0, 1000, 0})
      .addCustomer({3, 3, 0, 4, 0, 1000, 0})
      .addCustomer({4, 4, 0, 3, 0, 1000, 0})
      .addCustomer({5, 5, 0, 3, 0, 1000, 0})
      .build();
  }
};

struct create_exceeded_capacity_variant_2_problem_stream {
  std::stringstream operator()() {
    return SolomonBuilder()
      .setTitle("Exceeded capacity and two vehicles")
      .setVehicle(2, 10)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 1, 0, 3, 0, 1000, 10})
      .addCustomer({2, 2, 0, 3, 0, 1000, 10})
      .addCustomer({3, 3, 0, 3, 0, 1000, 10})
      .addCustomer({4, 4, 0, 2, 0, 1000, 10})
      .addCustomer({5, 5, 0, 2, 0, 1000, 10})
      .build();
  }
};

struct create_exceeded_time_problem_stream {
  std::stringstream operator()() {
    return SolomonBuilder()
      .setTitle("Exceeded time and two vehicles")
      .setVehicle(2, 10)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 1, 0, 1, 0, 1000, 10})
      .addCustomer({2, 2, 0, 1, 0, 1000, 10})
      .addCustomer({3, 3, 0, 1, 0, 1000, 10})
      .addCustomer({4, 4, 0, 1, 0, 1000, 10})
      .addCustomer({5, 100, 0, 2, 0, 101, 10})
      .build();
  }
};

struct create_c101_problem_stream {
  std::stringstream operator()() {
    return SolomonBuilder()
      .setTitle("Exceeded capacity and two vehicles")
      .setVehicle(25, 200)
      .addCustomer({0, 40, 50, 0, 0, 1236, 0})
      .addCustomer({1, 45, 68, 10, 912, 967, 90})
      .addCustomer({2, 45, 70, 30, 825, 870, 90})
      .addCustomer({3, 42, 66, 10, 65, 146, 90})
      .addCustomer({4, 42, 68, 10, 727, 782, 90})
      .addCustomer({5, 42, 65, 10, 15, 67, 90})
      .addCustomer({6, 40, 69, 20, 621, 702, 90})
      .addCustomer({7, 40, 66, 20, 170, 225, 90})
      .addCustomer({8, 38, 68, 20, 255, 324, 90})
      .addCustomer({9, 38, 70, 10, 534, 605, 90})
      .addCustomer({10, 35, 66, 10, 357, 410, 90})
      .addCustomer({11, 35, 69, 10, 448, 505, 90})
      .addCustomer({12, 25, 85, 20, 652, 721, 90})
      .addCustomer({13, 22, 75, 30, 30, 92, 90})
      .addCustomer({14, 22, 85, 10, 567, 620, 90})
      .addCustomer({15, 20, 80, 40, 384, 429, 90})
      .addCustomer({16, 20, 85, 40, 475, 528, 90})
      .addCustomer({17, 18, 75, 20, 99, 148, 90})
      .addCustomer({18, 15, 75, 20, 179, 254, 90})
      .addCustomer({19, 15, 80, 10, 278, 345, 90})
      .addCustomer({20, 30, 50, 10, 10, 73, 90})
      .addCustomer({21, 30, 52, 20, 914, 965, 90})
      .addCustomer({22, 28, 52, 20, 812, 883, 90})
      .addCustomer({23, 28, 55, 10, 732, 777, 90})
      .addCustomer({24, 25, 50, 10, 65, 144, 90})
      .addCustomer({25, 25, 52, 40, 169, 224, 90})
      .build();
  }
};

}  // namespace test
}  // namespace vrp

#endif  // VRP_TESTUTILS_PROBLEMSTREAMS_HPP