#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/convolutions/JointConvolutions.hpp"
#include "test_utils/TaskUtils.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>
#include <thrust/execution_policy.h>

using namespace vrp::algorithms::convolutions;
using namespace vrp::models;
using namespace vrp::utils;
using namespace vrp::test;

namespace {

template<typename T>
thrust::device_vector<T> create(const std::initializer_list<T>& list) {
  thrust::device_vector<T> data(list.begin(), list.end());
  return std::move(data);
}

Convolutions map(const std::vector<Convolution> convolutions, Pool& pool) {
  auto data = pool.acquire<thrust::device_vector<Convolution>>(convolutions.size());
  thrust::copy(convolutions.begin(), convolutions.end(), data->begin());
  return data;
}

void compare(const Convolution& left, const Convolution& right) {
  REQUIRE(left.demand == right.demand);
  REQUIRE(left.service == right.service);

  REQUIRE(left.customers.first == right.customers.first);
  REQUIRE(left.customers.second == right.customers.second);

  REQUIRE(left.times.first == right.times.first);
  REQUIRE(left.times.second == right.times.second);

  REQUIRE(left.tasks.first == right.tasks.first);
  REQUIRE(left.tasks.second == right.tasks.second);
}

void compare(const JointPair& left, const JointPair& right) {
  REQUIRE(left.rank == right.rank);
  REQUIRE(left.completeness == right.completeness);

  compare(left.pair.first, right.pair.first);
  compare(left.pair.second, right.pair.second);
}

};  // namespace

SCENARIO("Can create best convolution with 25 customers.", "[convolution][C101]") {
  int customers = 25 + 1;
  auto problem = Problem();
  auto tasks = Tasks(customers);
  problem.customers.demands = create({0,  10, 30, 10, 10, 10, 20, 20, 20, 10, 10, 10, 20,
                                      30, 10, 40, 40, 20, 20, 10, 10, 20, 20, 10, 10, 40});
  problem.customers.services = create({0,  90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                                       90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90});
  problem.customers.starts =
    create({0,  912, 825, 65,  727, 15,  621, 170, 255, 534, 357, 448, 652,
            30, 567, 384, 475, 99,  179, 278, 10,  914, 812, 732, 65,  169});
  tasks.ids = create(
    {0, 1, 20, 21, 22, 23, 2, 24, 25, 10, 11, 9, 6, 4, 5, 3, 7, 8, 12, 13, 17, 18, 19, 15, 16, 14});
  tasks.vehicles =
    create({0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6});
  tasks.costs = create<float>({0,  19, 10, 12, 12, 13, 36, 15, 17, 34, 37, 40, 43,
                               45, 15, 16, 18, 21, 42, 31, 35, 38, 43, 48, 53, 55});
  tasks.times = create({0,   1002, 100, 1004, 902, 822, 934, 155, 259, 447, 540, 633, 725,
                        817, 105,  196, 288,  380, 742, 120, 214, 307, 402, 497, 592, 684});
  Pool pool;

  auto convolutions = create_best_convolutions{}(problem, tasks, {0.5, 0.1, 1, pool});

  REQUIRE(convolutions->size() == 2);
  compare(convolutions->operator[](0), {0, 50, 367, {11, 4}, {448, 450}, {10, 13}});
  compare(convolutions->operator[](1), {0, 140, 560, {17, 14}, {99, 124}, {20, 25}});
}

SCENARIO("Can create joint convolution pair from two convolutions", "[convolution][join_pairs]") {
  int customers = 20 + 1;  // NOTE must be in sync with "right" definition below
  auto problem = Problem();
  auto tasks = Tasks(customers, 2);
  tasks.ids = create({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  Pool pool;
  std::vector<Convolution> left = {{0, 1, 1, {1, 4}, {}, {1, 4}},
                                   {0, 2, 2, {7, 13}, {}, {7, 13}},
                                   {0, 3, 3, {15, 19}, {}, {15, 19}}};
  std::vector<Convolution> right{{21, 4, 4, {1, 6}, {}, {1, 6}}, {21, 5, 5, {6, 11}, {}, {6, 11}}};


  auto result = create_joint_convolutions{}(problem, tasks, {0.2, 0.2, 1, pool}, map(left, pool),
                                            map(right, pool));


  compare(result->operator[](0), {4, static_cast<float>(6) / customers, {left.at(0), right.at(0)}});
  compare(result->operator[](1),
          {0, static_cast<float>(10) / customers, {left.at(0), right.at(1)}});

  compare(result->operator[](2),
          {0, static_cast<float>(13) / customers, {left.at(1), right.at(0)}});
  compare(result->operator[](3), {5, static_cast<float>(8) / customers, {left.at(1), right.at(1)}});

  compare(result->operator[](4),
          {0, static_cast<float>(11) / customers, {left.at(2), right.at(0)}});
  compare(result->operator[](5),
          {0, static_cast<float>(11) / customers, {left.at(2), right.at(1)}});
}
