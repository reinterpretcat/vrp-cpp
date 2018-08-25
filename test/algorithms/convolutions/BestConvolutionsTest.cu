#include "algorithms/convolutions/BestConvolutions.hpp"
#include "test_utils/ConvolutionUtils.hpp"
#include "test_utils/TaskUtils.hpp"
#include "test_utils/VectorUtils.hpp"

#include <catch/catch.hpp>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

using namespace vrp::algorithms::convolutions;
using namespace vrp::models;
using namespace vrp::runtime;
using namespace vrp::test;

namespace {

struct run_best_convolutions final {
  Solution::Shadow solution;
  Settings settings;
  vector_ptr<Convolution> output;

  EXEC_UNIT size_t operator()(int index) const {
    auto convolutions = create_best_convolutions{solution}(settings, index);
    auto source = *convolutions.data;
    thrust::copy(exec_unit_policy{}, source, source + convolutions.size, output);
    return convolutions.size;
  };

  EXEC_UNIT size_t operator()(size_t left, size_t right) const { return right; }
};

Problem createBasicProblem() {
  auto problem = Problem();
  problem.customers.demands = create({0,  10, 30, 10, 10, 10, 20, 20, 20, 10, 10, 10, 20,
                                      30, 10, 40, 40, 20, 20, 10, 10, 20, 20, 10, 10, 40});
  problem.customers.services = create({0,  90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                                       90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90});
  problem.customers.starts =
    create({0,  912, 825, 65,  727, 15,  621, 170, 255, 534, 357, 448, 652,
            30, 567, 384, 475, 99,  179, 278, 10,  914, 812, 732, 65,  169});
  return std::move(problem);
}

Solution createBasicSolution() {
  int customers = 25 + 1;
  auto problem = createBasicProblem();
  auto tasks = Tasks(customers);

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

SCENARIO("Can create best convolutions with 25 customers with two convolutions",
         "[best_convolutions][C101]") {
  size_t size = 2;
  auto solution = createBasicSolution();
  auto output = vector<Convolution>(size);

  auto runner = run_best_convolutions{solution.getShadow(), {0.75, 3}, output.data()};
  auto result = thrust::transform_reduce(exec_unit, thrust::make_counting_iterator(0),
                                         thrust::make_counting_iterator(1), runner, 0, runner);
  REQUIRE(result == size);
  compare(output[0], {0, 50, 367, {11, 4}, {450, 450}, {10, 13}});
  compare(output[1], {0, 140, 560, {17, 14}, {124, 124}, {20, 25}});
}

SCENARIO("Can create best convolutions with 25 customers with three convolutions",
         "[best_convolutions][C101]") {
  size_t size = 3;
  auto solution = createBasicSolution();
  auto output = vector<Convolution>(size);

  auto runner = run_best_convolutions{solution.getShadow(), {0.9, 3}, output.data()};
  auto result = thrust::transform_reduce(exec_unit, thrust::make_counting_iterator(0),
                                         thrust::make_counting_iterator(1), runner, 0, runner);
  REQUIRE(result == 3);
  compare(output[0], {0, 100, 648, {25, 4}, {169, 169}, {8, 13}});
  compare(output[1], {0, 70, 636, {3, 12}, {106, 106}, {15, 18}});
  compare(output[2], {0, 140, 560, {17, 14}, {124, 124}, {20, 25}});
}
