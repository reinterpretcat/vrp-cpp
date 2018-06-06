#include "algorithms/convolutions/JointConvolutions.hpp"
#include "test_utils/ConvolutionUtils.hpp"
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

/*
namespace {
Convolutions map(const std::vector<Convolution> convolutions, Pool& pool) {
  auto data = pool.acquire<thrust::device_vector<Convolution>>(convolutions.size());
  thrust::copy(convolutions.begin(), convolutions.end(), data->begin());
  return data;
}
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
*/