#pragma once

#include "test_utils/streams/SolomonBuilder.hpp"

namespace vrp::test {

/// Specifies problem with 5 sequential customers, demand equals 1,
/// and permissive time windows.
struct create_sequential_problem_stream {
  std::stringstream operator()(int vehicles = 1, int capacity = 10) {
    return SolomonBuilder()
      .setTitle("Sequential customers")
      .setVehicle(vehicles, capacity)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 1, 0, 1, 0, 1000, 10})
      .addCustomer({2, 2, 0, 1, 0, 1000, 10})
      .addCustomer({3, 3, 0, 1, 0, 1000, 10})
      .addCustomer({4, 4, 0, 1, 0, 1000, 10})
      .addCustomer({5, 5, 0, 1, 0, 1000, 10})
      .build();
  }
};

/// Specifies problem with 5 sequential customers, demand equals 1,
/// and one strict time window.
struct create_time_problem_stream {
  std::stringstream operator()(int vehicles = 2, int capacity = 10) {
    return SolomonBuilder()
      .setTitle("Sequential customers")
      .setVehicle(vehicles, capacity)
      .addCustomer({0, 0, 0, 0, 0, 1000, 0})
      .addCustomer({1, 1, 0, 1, 0, 1000, 10})
      .addCustomer({2, 2, 0, 1, 0, 1000, 10})
      .addCustomer({3, 3, 0, 1, 0, 1000, 10})
      .addCustomer({4, 4, 0, 1, 0, 1000, 10})
      .addCustomer({5, 5, 0, 1, 0, 10, 10})
      .build();
  }
};

/// Specifies the problem from solomon set.
struct create_c101_25_problem_stream {
  std::stringstream operator()(int vehicles = 25, int capacity = 200) {
    return SolomonBuilder()
      .setTitle("c101_25 problem")
      .setVehicle(vehicles, capacity)
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
}