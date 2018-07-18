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
using namespace vrp::runtime;
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
    EXEC_UNIT Tuple operator()(const Tuple& left, const Tuple& right) const {
      // NOTE use big number for cost as barrier between vehicles
      const float CostBarrier = 3.4E37;
      return thrust::get<0>(left) == thrust::get<0>(right)
               ? thrust::make_tuple(thrust::get<0>(left),
                                    thrust::get<1>(left) - thrust::get<1>(right))
               : thrust::make_tuple(thrust::get<0>(left), CostBarrier);
    }
  };

  /// Maps cost gradient info to cost.
  struct map_gradient final {
    template<typename Tuple>
    EXEC_UNIT float operator()(const Tuple& tuple) const {
      return thrust::get<1>(tuple);
    }
  };

  EXEC_UNIT void operator()(const Tasks::Shadow& tasks, vector_ptr<float> differences) const {
    thrust::adjacent_difference(
      exec_unit_policy{},
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

  EXEC_UNIT void operator()(const vector_ptr<int> vehicles,
                            vector_ptr<float> differences,
                            vector_ptr<float> medians,
                            vector_ptr<bool> plan) const {
    // initialize medians
    thrust::copy(exec_unit_policy{}, differences, differences + size, medians);

    // sort and get median
    thrust::sort(exec_unit_policy{}, medians, medians + size);
    float median = medians[(size - vehicles[size - 1]) * medianRatio];

    // create plan using median
    thrust::transform(exec_unit_policy{}, differences, differences + size, plan, _1 <= median);
  }
};

/// Estimates convolutions based on partial plan.
struct estimate_convolutions final {
  size_t size;
  /// Convolution operator.
  struct compare_plan final {
    template<typename Tuple>
    EXEC_UNIT bool operator()(const Tuple& left, const Tuple& right) const {
      return thrust::get<0>(left) == thrust::get<0>(right);
    }
  };

  EXEC_UNIT size_t operator()(const vector_ptr<bool> plan,
                              vector_ptr<thrust::tuple<bool, int>> output,
                              vector_ptr<int> lengths) const {
    auto result = thrust::reduce_by_key(
      exec_unit_policy{},
      thrust::make_zip_iterator(thrust::make_tuple(plan, thrust::make_counting_iterator(0))),
      thrust::make_zip_iterator(
        thrust::make_tuple(plan + size, thrust::make_counting_iterator(static_cast<int>(size)))),
      thrust::constant_iterator<int>(1), output, lengths, compare_plan());

    return static_cast<size_t>(result.first - output);
  }
};

/// Creates convolutions based on estimation.
struct create_convolutions final {
  Solution::Shadow solution;
  float convolutionRatio;
  size_t size;
  size_t groups;

  /// Filters group by plan and length.
  struct filter_group final {
    int limit;
    template<typename Tuple>
    EXEC_UNIT bool operator()(const Tuple& tuple) const {
      return !(thrust::get<0>(thrust::get<0>(tuple)) && thrust::get<1>(tuple) > limit);
    }
  };

  /// Maps group to its convolution representation.
  struct map_group final {
    Solution::Shadow solution;
    int base;

    EXEC_UNIT int operator()(int id) const { return *(solution.problem.customers.demands + id); }

    EXEC_UNIT Convolution operator()(const thrust::tuple<thrust::tuple<bool, int>, int>& tuple) {
      auto problem = solution.problem;
      auto tasks = solution.tasks;

      int seq = thrust::get<1>(thrust::get<0>(tuple));
      int length = thrust::get<1>(tuple);

      auto start = base + seq;
      auto end = start + length - 1;
      auto firstCustomerService = problem.customers.services[tasks.ids[start]];

      auto demand = thrust::transform_reduce(exec_unit_policy{}, tasks.ids + start,
                                             tasks.ids + end + 1, *this, 0, thrust::plus<int>());
      return Convolution{base, demand,
                         // new service time from total duration
                         tasks.times[end] - tasks.times[start] + firstCustomerService,
                         // get fist and last customer
                         thrust::make_pair<int, int>(tasks.ids[start], tasks.ids[end]),
                         // get TW which is [first customer TW start, first customer ETA]
                         thrust::make_pair<int, int>(problem.customers.starts[tasks.ids[start]],
                                                     tasks.times[start] - firstCustomerService),
                         // calculate task range (all inclusive)
                         thrust::make_pair<int, int>(start - base, end - base)};
    };
  };

  EXEC_UNIT size_t operator()(int base,
                              const vector_ptr<thrust::tuple<bool, int>> output,
                              const vector_ptr<int> lengths,
                              vector_ptr<Convolution> convolutions) const {
    auto limit = __float2int_rn(solution.problem.size * convolutionRatio);

    auto newEnd = thrust::remove_copy_if(
      exec_unit_policy{}, thrust::make_zip_iterator(thrust::make_tuple(output, lengths)),
      thrust::make_zip_iterator(thrust::make_tuple(output + groups, lengths + groups)),
      thrust::make_transform_output_iterator(convolutions, map_group{solution, base}),
      filter_group{limit});

    // TODO simplify this
    return static_cast<size_t>(thrust::distance(
      thrust::make_transform_output_iterator(convolutions, map_group{solution, base}), newEnd));
  }
};

}  // namespace

EXEC_UNIT Convolutions create_best_convolutions::operator()(const Settings& settings,
                                                            int index) const {
  auto size = static_cast<size_t>(solution.problem.size);
  auto begin = index * size;
  auto end = begin + size;

  auto differences = make_unique_ptr_data<float>(size);
  create_cost_differences{begin, end}.operator()(solution.tasks, *differences);

  auto medians = make_unique_ptr_data<float>(size);
  auto plan = make_unique_ptr_data<bool>(size);
  create_partial_plan{settings.MedianRatio, size}.operator()(solution.tasks.vehicles, *differences,
                                                             *medians, *plan);

  auto output = make_unique_ptr_data<thrust::tuple<bool, int>>(size);
  auto lengths = make_unique_ptr_data<int>(size);
  auto groups = estimate_convolutions{size}.operator()(*plan, *output, *lengths);

  auto convolutions = make_unique_ptr_data<Convolution>(size);
  auto resultSize =
    create_convolutions{solution, settings.ConvolutionRatio, size, groups}.operator()(
      static_cast<int>(begin), *output, *lengths, *convolutions);

  return {resultSize, std::move(convolutions)};
}
