#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/convolutions/JointConvolutions.hpp"
#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "test_utils/MemoryUtils.hpp"
#include "test_utils/TaskUtils.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

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

/*JointPairs createPairs(Pool& pool,
                       const thrust::pair<size_t, size_t>& dimens,
                       const std::initializer_list<JointPair>& list) {
  auto pairs = pool.acquire<thrust::device_vector<JointPair>>(list.size());
  thrust::copy(thrust::device, list.begin(), list.end(), pairs->begin());
  return {dimens, std::move(pairs)};
}


Convolutions map(const std::vector<Convolution> convolutions, Pool& pool) {
  auto data = pool.acquire<thrust::device_vector<Convolution>>(convolutions.size());
  thrust::copy(convolutions.begin(), convolutions.end(), data->begin());
  return data;
}*/

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
  REQUIRE(left.similarity == right.similarity);
  REQUIRE(left.completeness == right.completeness);

  compare(left.pair.first, right.pair.first);
  compare(left.pair.second, right.pair.second);
}

using ConvolutionResult = thrust::pair<size_t, Convolutions::ConvolutionsPtr>;

struct run_best_convolutions final {
  Solution::Shadow solution;
  DevicePool::Pointer pool;
  Settings settings;

  __device__ ConvolutionResult operator()(int index) const {
    printf("run_best_convolutions..\n");
    auto convolutions = create_best_convolutions{solution, pool}(settings, index);
    return {convolutions.size, *convolutions.data.release()};
  };

  __device__ ConvolutionResult operator()(const ConvolutionResult& left,
                                          const ConvolutionResult& right) const {
    return right;
  }
};

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
  auto pool = DevicePool::create(1, 4, static_cast<size_t>(customers));
  Solution solution(std::move(problem), std::move(tasks));

  auto runner = run_best_convolutions{solution.getShadow(), getPool(), {0.5, 0.1}};
  auto result = thrust::transform_reduce(thrust::device, thrust::make_counting_iterator(0),
                                         thrust::make_counting_iterator(1), runner,
                                         ConvolutionResult{}, runner);
  printf("Done!\n");
  REQUIRE(result.first == 2);
  // compare(convolutions->operator[](0), {0, 50, 367, {11, 4}, {448, 450}, {10, 13}});
  // compare(convolutions->operator[](1), {0, 140, 560, {17, 14}, {99, 124}, {20, 25}});
}

/*SCENARIO("Can create joint convolution pair from two convolutions", "[convolution][join_pairs]") {
  int customers = 20 + 1;
  auto problem = Problem();
  auto tasks = Tasks(customers, 2);
  tasks.ids = create({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  Pool pool;
  std::vector<Convolution> left = {{0, 1, 1, {1, 4}, {}, {1, 4}},
                                   {0, 2, 2, {7, 13}, {}, {7, 13}},
                                   {0, 3, 3, {15, 19}, {}, {15, 19}}};
  std::vector<Convolution> right{{customers, 4, 4, {1, 6}, {}, {1, 6}},
                                 {customers, 5, 5, {6, 11}, {}, {6, 11}}};
  Solution solution(std::move(problem), std::move(tasks));


  auto result =
    create_joint_convolutions{}(solution, {0.2, 0.2, pool}, map(left, pool), map(right, pool));


  REQUIRE(result.dimens.first == 3);
  REQUIRE(result.dimens.second == 2);

  compare(result.pairs->operator[](0), {4, 6, {left.at(0), right.at(0)}});
  compare(result.pairs->operator[](1), {0, 10, {left.at(0), right.at(1)}});

  compare(result.pairs->operator[](2), {0, 13, {left.at(1), right.at(0)}});
  compare(result.pairs->operator[](3), {5, 8, {left.at(1), right.at(1)}});

  compare(result.pairs->operator[](4), {0, 11, {left.at(2), right.at(0)}});
  compare(result.pairs->operator[](5), {0, 11, {left.at(2), right.at(1)}});
}

SCENARIO("Can create sliced convolutions from joint pairs", "[convolution][sliced]") {
  Pool pool;
  int customers = 20 + 1;
  auto problem = Problem();
  auto tasks = Tasks(customers, 2);
  auto settings = Settings{0.2, 0.2, pool};
  tasks.ids = create({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

  std::vector<Convolution> left = {{0, 1, 1, {1, 4}, {}, {1, 4}},
                                   {0, 2, 2, {7, 13}, {}, {7, 13}},
                                   {0, 3, 3, {15, 19}, {}, {15, 19}}};
  std::vector<Convolution> right{{customers, 4, 4, {1, 6}, {}, {1, 6}},
                                 {customers, 5, 5, {6, 11}, {}, {6, 11}}};
  auto pairs = createPairs(pool, {3, 2},
                           {{4, 6, {left.at(0), right.at(0)}},
                            {0, 10, {left.at(0), right.at(1)}},
                            {0, 13, {left.at(1), right.at(0)}},
                            {5, 8, {left.at(1), right.at(1)}},
                            {0, 11, {left.at(2), right.at(0)}},
                            {0, 11, {left.at(2), right.at(1)}}});
  Solution solution(std::move(problem), std::move(tasks));

  auto result = create_sliced_convolutions{}(solution, settings, pairs);

  REQUIRE(result->size() == (left.size() + right.size()));
  compare(result->operator[](0), left.at(0));
  compare(result->operator[](1), right.at(1));
  compare(result->operator[](2), left.at(1));
  compare(result->operator[](3), right.at(0));
  compare(result->operator[](4), left.at(2));
}*/
