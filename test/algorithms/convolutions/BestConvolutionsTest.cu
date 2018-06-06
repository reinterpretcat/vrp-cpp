#include "algorithms/convolutions/BestConvolutions.hpp"
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

namespace {

using ConvolutionResult = thrust::pair<size_t, Convolutions::ConvolutionsPtr>;

struct run_best_convolutions final {
  Solution::Shadow solution;
  DevicePool::Pointer pool;
  Settings settings;

  __device__ ConvolutionResult operator()(int index) const {
    auto convolutions = create_best_convolutions{solution, pool}(settings, index);
    return {convolutions.size, *convolutions.data.release()};
  };

  __device__ ConvolutionResult operator()(const ConvolutionResult& left,
                                          const ConvolutionResult& right) const {
    return right;
  }
};

Solution createBasicSolution() {
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

  return {std::move(problem), std::move(tasks)};
}

}  // namespace

SCENARIO("Can create best convolutions with 25 customers with convolution ratio 0.1", "[convolution][C101]") {
  auto solution = createBasicSolution();

  auto runner = run_best_convolutions{solution.getShadow(), getPool(), {0.5, 0.1}};
  auto result = thrust::transform_reduce(thrust::device, thrust::make_counting_iterator(0),
                                         thrust::make_counting_iterator(1), runner,
                                         ConvolutionResult{}, runner);
  REQUIRE(result.first == 2);
  // compare(convolutions->operator[](0), {0, 50, 367, {11, 4}, {448, 450}, {10, 13}});
  // compare(convolutions->operator[](1), {0, 140, 560, {17, 14}, {99, 124}, {20, 25}});
}
