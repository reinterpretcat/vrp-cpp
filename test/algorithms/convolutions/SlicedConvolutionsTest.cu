#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/convolutions/JointConvolutions.hpp"
#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "test_utils/ConvolutionUtils.hpp"
#include "test_utils/MemoryUtils.hpp"
#include "test_utils/TaskUtils.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <utils/memory/Allocations.hpp>

using namespace vrp::algorithms::convolutions;
using namespace vrp::models;
using namespace vrp::utils;
using namespace vrp::test;

namespace {
const int customers = 20 + 1;

using ConvolutionResult = thrust::pair<size_t, thrust::device_ptr<Convolution>>;

/// Creates solution skeleton.
Solution createBasicSolution() {
  auto problem = Problem();
  auto tasks = Tasks(customers, 2);
  tasks.ids = create({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

  return {std::move(problem), std::move(tasks)};
}

/// Prepares and runs testing function on device.
struct run_sliced_convolutions final {
  Solution::Shadow solution;
  DevicePool::Pointer pool;
  Settings settings;
  thrust::pair<size_t, size_t> dimens;
  thrust::device_ptr<JointPair> data;

  thrust::device_ptr<ConvolutionResult> result;

  __device__ void operator()(int index) {
    auto jointPairs = pool.get()->jointPairs(customers);
    thrust::copy(thrust::device, data, data + dimens.first * dimens.second, *jointPairs);

    auto sliced =
      create_sliced_convolutions{solution, pool}({0.2, 0.2}, {dimens, std::move(jointPairs)});

    thrust::copy(thrust::device, *sliced.data, *sliced.data + sliced.size, result.get()->second);
    result.get()->first = sliced.size;
  }
};

/// Calculates results with handling memory transfer between host and device
thrust::host_vector<Convolution> getResult(Solution& solution,
                                           const thrust::pair<size_t, size_t>& dimens,
                                           const std::initializer_list<JointPair>& list) {
  const int expected = 5;
  thrust::device_vector<Convolution> resultData(expected);
  thrust::device_vector<JointPair> jointPairs(list.begin(), list.end());

  auto runner = run_sliced_convolutions{
    solution.getShadow(), getPool(),
    {0.2, 0.2},           dimens,
    jointPairs.data(),    allocate<ConvolutionResult>({0, resultData.data()})};

  thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1), runner);

  auto result = release(runner.result);
  thrust::host_vector<Convolution> data(result.second, result.second + result.first);

  return std::move(data);
}

}  // namespace

SCENARIO("Can create sliced convolutions from joint pairs", "[convolution][sliced]") {
  auto solution = createBasicSolution();

  std::vector<Convolution> left = {{0, 1, 1, {1, 4}, {}, {1, 4}},
                                   {0, 2, 2, {7, 13}, {}, {7, 13}},
                                   {0, 3, 3, {15, 19}, {}, {15, 19}}};
  std::vector<Convolution> right{{customers, 4, 4, {1, 6}, {}, {1, 6}},
                                 {customers, 5, 5, {6, 11}, {}, {6, 11}}};

  auto result = getResult(solution, {3, 2},
                          {{4, 6, {left.at(0), right.at(0)}},
                           {0, 10, {left.at(0), right.at(1)}},
                           {0, 13, {left.at(1), right.at(0)}},
                           {5, 8, {left.at(1), right.at(1)}},
                           {0, 11, {left.at(2), right.at(0)}},
                           {0, 11, {left.at(2), right.at(1)}}});

  REQUIRE(result.size() == 5);
  compare(result[0], left.at(0));
  compare(result[1], right.at(1));
  compare(result[2], left.at(1));
  compare(result[3], right.at(0));
  compare(result[4], left.at(2));
}
