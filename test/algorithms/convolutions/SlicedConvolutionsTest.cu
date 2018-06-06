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

/*namespace {
JointPairs createPairs(Pool& pool,
                       const thrust::pair<size_t, size_t>& dimens,
                       const std::initializer_list<JointPair>& list) {
  auto pairs = pool.acquire<thrust::device_vector<JointPair>>(list.size());
  thrust::copy(thrust::device, list.begin(), list.end(), pairs->begin());
  return {dimens, std::move(pairs)};
}
};  // namespace

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
