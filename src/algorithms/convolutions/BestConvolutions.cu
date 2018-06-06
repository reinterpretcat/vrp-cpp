#include "algorithms/convolutions/BestConvolutions.hpp"
#include "iterators/CartesianProduct.hpp"

#include <cmath>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

using namespace vrp::algorithms::convolutions;
using namespace vrp::models;
using namespace vrp::iterators;
using namespace vrp::utils;
using namespace thrust::placeholders;

namespace {
/// Creates cost differences (gradients) as cost change
/// between two customers.
struct create_cost_differences final {
  /// Base index.
  size_t begin;
  /// Last index.
  size_t end;

  /// Creates a gradient of costs between tasks.
  struct make_gradient final {
    template<typename Tuple>
    __device__ Tuple operator()(const Tuple& left, const Tuple& right) const {
      return thrust::get<0>(left) == thrust::get<0>(right)
               ? thrust::make_tuple(thrust::get<0>(left),
                                    thrust::get<1>(left) - thrust::get<1>(right))
               : left;
    }
  };

  /// Maps cost gradient info to cost.
  struct map_gradient final {
    template<typename Tuple>
    __device__ float operator()(const Tuple& tuple) const {
      return thrust::get<1>(tuple);
    }
  };

  __device__ void operator()(const Tasks::Shadow& tasks,
                             thrust::device_ptr<float> differences) const {
    thrust::adjacent_difference(
      thrust::device,
      thrust::make_zip_iterator(thrust::make_tuple(tasks.vehicles + begin, tasks.costs + begin)),
      thrust::make_zip_iterator(thrust::make_tuple(tasks.vehicles + end, tasks.costs + end)),
      thrust::make_transform_output_iterator(differences, map_gradient()), make_gradient());
  }
};

/// Creates partial plan taking into account median of
/// cost differences: customer is served only if cost change
/// is lower than median value.
struct create_partial_plan final {
  float medianRatio;
  size_t size;

  __device__ void operator()(thrust::device_ptr<float> differences,
                             thrust::device_ptr<float> medians,
                             thrust::device_ptr<bool> plan) const {
    // initialize medians
    thrust::copy(thrust::device, differences, differences + size, medians);

    // sort and get median
    thrust::sort(thrust::device, medians, medians + size);
    float median = medians[size * medianRatio];

    // create plan using median
    thrust::transform(thrust::device, differences, differences + size, plan, _1 <= median);
  }
};

/// Estimates convolutions based on partial plan.
struct estimate_convolutions final {
  size_t size;
  /// Convolution operator.
  struct compare_plan final {
    template<typename Tuple>
    __device__ bool operator()(const Tuple& left, const Tuple& right) const {
      return thrust::get<0>(left) == thrust::get<0>(right);
    }
  };

  __device__ void operator()(const thrust::device_ptr<bool> plan,
                               thrust::device_ptr<thrust::tuple<bool, int>> output,
                               thrust::device_ptr<int> lengths) const {
    thrust::reduce_by_key(
      thrust::device,
      thrust::make_zip_iterator(thrust::make_tuple(plan, thrust::make_counting_iterator(0))),
      thrust::make_zip_iterator(
        thrust::make_tuple(plan + size, thrust::make_counting_iterator(static_cast<int>(size)))),
      thrust::constant_iterator<int>(1), output, lengths, compare_plan());
  }
};

/// Creates convolutions based on estimation.
struct create_convolutions final {
  Solution::Shadow solution;
  float convolutionRatio;
  size_t size;

  /// Filters group by plan and length.
  struct filter_group final {
    int limit;
    template<typename Tuple>
    __device__ bool operator()(const Tuple& tuple) const {
      return !(thrust::get<0>(thrust::get<0>(tuple)) && thrust::get<1>(tuple) > limit);
    }
  };

  /// Maps group to its convolution representation.
  struct map_group final {
    Solution::Shadow solution;
    int base;

    __device__ int operator()(int id) const { return *(solution.problem.customers.demands + id); }

    __device__ Convolution operator()(const thrust::tuple<thrust::tuple<bool, int>, int>& tuple) {
      auto problem = solution.problem;
      auto tasks = solution.tasks;

      int seq = thrust::get<1>(thrust::get<0>(tuple));
      int length = thrust::get<1>(tuple);

      auto end = base + seq + 1;
      auto start = end - length;
      auto firstCustomerService = problem.customers.services[tasks.ids[start]];

      return Convolution{// base index
                         base,
                         // total customers demand
                         thrust::transform_reduce(thrust::device, tasks.ids + start,
                                                  tasks.ids + end, *this, 0, thrust::plus<int>()),
                         // new service time from total duration
                         tasks.times[end - 1] - tasks.times[start] + firstCustomerService,

                         // get fist and last customer
                         thrust::make_pair<int, int>(tasks.ids[start], tasks.ids[end - 1]),
                         // get TW which is [first customer TW start, first customer ETA]
                         thrust::make_pair<int, int>(problem.customers.starts[tasks.ids[start]],
                                                     tasks.times[start] - firstCustomerService),
                         // calculate task range (all inclusive)
                         thrust::make_pair<int, int>(start - base, end - base - 1)};
    };
  };

  __device__ size_t operator()(int base,
                               const thrust::device_ptr<thrust::tuple<bool, int>> output,
                               const thrust::device_ptr<int> lengths,
                               thrust::device_ptr<Convolution> convolutions) const {
    auto limit = __float2int_rn(solution.problem.size * convolutionRatio);

    auto newEnd = thrust::remove_copy_if(
      thrust::device, thrust::make_zip_iterator(thrust::make_tuple(output, lengths)),
      thrust::make_zip_iterator(thrust::make_tuple(output + size, lengths + size)),
      thrust::make_transform_output_iterator(convolutions, map_group{solution, base}),
      filter_group{limit});

    // TODO simplify this
    return static_cast<size_t>(thrust::distance(
      thrust::make_transform_output_iterator(convolutions, map_group{solution, 0}), newEnd));
  }
};

}  // namespace

__device__ Convolutions create_best_convolutions::operator()(const Settings& settings,
                                                             int index) const {
  auto size = static_cast<size_t>(solution.problem.size);
  auto begin = index * size;
  auto end = begin + size;

  auto differences = pool.get()->floats(size);
  create_cost_differences{begin, end}.operator()(solution.tasks, *differences);

  auto medians = pool.get()->floats(size);
  auto plan = pool.get()->bools(size);
  create_partial_plan{settings.MedianRatio, size}.operator()(*differences, *medians, *plan);

  auto output = pool.get()->boolInts(size);
  auto lengths = pool.get()->ints(size);
  estimate_convolutions{size}.operator()(*plan, *output, *lengths);

  auto convolutions = pool.get()->convolutions(size);
  auto resultSize =
    create_convolutions{solution, settings.ConvolutionRatio, size}.operator()(
      static_cast<int>(begin), *output, *lengths, *convolutions);

  return {resultSize, std::move(convolutions)};
}
