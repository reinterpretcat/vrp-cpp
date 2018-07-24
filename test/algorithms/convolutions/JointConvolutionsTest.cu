#include "algorithms/convolutions/JointConvolutions.hpp"
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
const int customers = 20 + 1;

using JoinPairResult = thrust::pair<thrust::pair<size_t, size_t>, vector_ptr<JointPair>>;
using Result = thrust::pair<thrust::pair<size_t, size_t>, thrust::host_vector<JointPair>>;

/// Minimal solution to launch testing function.
Solution createBasicSolution() {
  auto problem = Problem();
  auto tasks = Tasks(customers, 2);
  tasks.ids = create({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

  return {std::move(problem), std::move(tasks)};
}

/// Prepares and runs testing function on device or host.
struct run_joint_convolutions final {
  using ConvolutionData = thrust::pair<size_t, vector_ptr<Convolution>>;

  Solution::Shadow solution;
  Settings settings;
  ConvolutionData left;
  ConvolutionData right;

  vector_ptr<JoinPairResult> result;

  EXEC_UNIT void operator()(int index) {
    // TODO simplify this
    auto leftPooled = make_unique_ptr_data<Convolution>(static_cast<size_t>(customers));
    thrust::copy(exec_unit_policy{}, left.second, left.second + left.first, *leftPooled);
    auto rightPooled = make_unique_ptr_data<Convolution>(static_cast<size_t>(customers));
    thrust::copy(exec_unit_policy{}, right.second, right.second + right.first, *rightPooled);

    auto joinPairs = create_joint_convolutions{solution}(
      settings, {left.first, std::move(leftPooled)}, {right.first, std::move(rightPooled)});

    auto resultPtr = raw_pointer_cast(result);

    thrust::copy(exec_unit_policy{}, *joinPairs.data,
                 *joinPairs.data + joinPairs.dimens.first * joinPairs.dimens.second,
                 resultPtr->second);

    resultPtr->first = joinPairs.dimens;
  };
};

/// Calculates results with handling memory transfer between host and device
Result getResult(Solution& solution,
                 const std::vector<Convolution>& left,
                 const std::vector<Convolution>& right) {
  const int expected = 6;
  vector<JointPair> resultData(expected);
  vector<Convolution> leftDev(left.begin(), left.end());
  vector<Convolution> rightDev(right.begin(), right.end());

  auto runner =
    run_joint_convolutions{solution.getShadow(),
                           {0.75, 0.1},
                           {leftDev.size(), leftDev.data()},
                           {rightDev.size(), rightDev.data()},
                           vrp::runtime::allocate<JoinPairResult>({{0, 0}, resultData.data()})};
  thrust::for_each(exec_unit, thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
                   runner);
  auto result = vrp::runtime::release<JoinPairResult>(runner.result);
  thrust::host_vector<JointPair> data(result.second, result.second + expected);

  return {result.first, std::move(data)};
}

}  // namespace

SCENARIO("Can create joint convolution pair from two convolutions", "[convolution][join_pairs]") {
  Solution solution = createBasicSolution();
  std::vector<Convolution> left = {{0, 1, 1, {1, 4}, {}, {1, 4}},
                                   {0, 2, 2, {7, 13}, {}, {7, 13}},
                                   {0, 3, 3, {15, 19}, {}, {15, 19}}};
  std::vector<Convolution> right = {{customers, 4, 4, {1, 6}, {}, {1, 6}},
                                    {customers, 5, 5, {6, 11}, {}, {6, 11}}};


  auto result = getResult(solution, left, right);


  REQUIRE(result.first.first == 3);
  REQUIRE(result.first.second == 2);

  compare(result.second[0], {4, 6, {left.at(0), right.at(0)}});
  compare(result.second[1], {0, 10, {left.at(0), right.at(1)}});

  compare(result.second[2], {0, 13, {left.at(1), right.at(0)}});
  compare(result.second[3], {5, 8, {left.at(1), right.at(1)}});

  compare(result.second[4], {0, 11, {left.at(2), right.at(0)}});
  compare(result.second[5], {0, 11, {left.at(2), right.at(1)}});
}
