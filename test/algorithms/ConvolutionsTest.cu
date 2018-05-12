#include <catch/catch.hpp>

#include "algorithms/Convolutions.hpp"

using namespace vrp::algorithms;
using namespace vrp::models;
using namespace vrp::utils;

namespace {

template <typename T>
thrust::device_vector<T> create(const std::initializer_list<T> &list) {
  thrust::device_vector<T> data (list.begin(), list.end());
  return std::move(data);
}

void compare(const Convolution &left, const Convolution &right) {
  REQUIRE(left.demand == right.demand);
  REQUIRE(left.service == right.service);

  REQUIRE(left.customers.first == right.customers.first);
  REQUIRE(left.customers.second == right.customers.second);

  REQUIRE(left.times.first == right.times.first);
  REQUIRE(left.times.second == right.times.second);

  REQUIRE(left.tasks.first == right.tasks.first);
  REQUIRE(left.tasks.second == right.tasks.second);
}

}

SCENARIO("Can create convolution from C101 solution with 25 customers.", "[convolution]") {
  int customers = 25 + 1;
  auto tasks = Tasks(customers);
  auto problem = Problem();
  problem.customers.demands = create({0, 10, 30, 10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 30, 10, 40, 40, 20, 20, 10, 10, 20, 20, 10, 10, 40});
  problem.customers.services = create({0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90});
  problem.customers.starts = create({0, 912, 825, 65, 727, 15, 621, 170, 255, 534, 357, 448, 652, 30, 567, 384, 475, 99, 179, 278, 10, 914, 812, 732, 65, 169});
  tasks.ids = create({0, 1, 20, 21, 22, 23, 2, 24, 25, 10, 11, 9, 6, 4, 5, 3, 7, 8, 12, 13, 17, 18, 19, 15, 16, 14});
  tasks.vehicles = create({0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6});
  tasks.costs = create<float>({0, 19, 10, 12, 12, 13, 36, 15, 17, 34, 37, 40, 43, 45, 15, 16, 18, 21, 42, 31, 35, 38, 43, 48, 53, 55});
  tasks.times = create({0, 1002, 100, 1004, 902, 822, 934, 155, 259, 447, 540, 633, 725, 817, 105, 196, 288, 380, 742, 120, 214, 307, 402, 497, 592, 684});
  Pool pool;

  auto convolutions = create_best_convolutions {} (problem, tasks, {0.5, 0.1, 1}, pool);

  REQUIRE(convolutions->size() == 2);
  compare(convolutions->operator[](0), {50, 367, {11, 4}, {448, 450}, {10, 13}});
  compare(convolutions->operator[](1), {140, 560, {17, 14}, {99, 124}, {20, 25}});
}
