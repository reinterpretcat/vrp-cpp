#include "algorithms/common/Convolutions.hpp"
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

/// Contains information about route leg: vehicle, its cost, and
/// is it good for convolution or not (plan).
using CostRouteLeg = thrust::tuple<int, float, bool>;

/// Creates cost differences (gradients) as cost change
/// between two customers.
struct create_cost_legs final {
  /// Base index.
  size_t begin;
  /// Last index.
  size_t end;

  /// Creates a gradient of costs between tasks.
  struct make_gradient final {
    template<typename Tuple>
    EXEC_UNIT CostRouteLeg operator()(const Tuple& left, const Tuple& right) const {
      auto cost = thrust::get<0>(left) == thrust::get<0>(right)
                    ? thrust::get<1>(left) - thrust::get<1>(right)
                    : thrust::get<1>(left);
      return {thrust::get<0>(left), cost, false};
    }
  };

  EXEC_UNIT void operator()(const Tasks::Shadow& tasks, vector_ptr<CostRouteLeg> legs) const {
    thrust::adjacent_difference(
      exec_unit_policy{},
      thrust::make_zip_iterator(thrust::make_tuple(tasks.vehicles + begin, tasks.costs + begin,
                                                   thrust::make_constant_iterator(false))),
      thrust::make_zip_iterator(thrust::make_tuple(tasks.vehicles + end, tasks.costs + end,
                                                   thrust::make_constant_iterator(false))),
      legs, make_gradient{});
  }
};

/// Fills leg plan taking into account median of cost differences: customer is
/// served only if cost change is lower than median value.
struct fill_leg_plan final {
  float medianRatio;
  int begin;
  size_t size;

  EXEC_UNIT void operator()(const vector_ptr<CostRouteLeg> legs, vector_ptr<float> medians) const {
    // initialize medians
    thrust::transform(exec_unit_policy{}, legs, legs + size, medians, extract_cost{});

    // sort and get median
    thrust::sort(exec_unit_policy{}, medians, medians + size);
    float median = medians[static_cast<int>((size - 1) * medianRatio)];

    // create plan using median
    thrust::for_each(exec_unit_policy{}, legs, legs + size, prepare_plan{median});
  }

  struct extract_cost final {
    EXEC_UNIT float operator()(const CostRouteLeg& leg) { return thrust::get<1>(leg); }
  };

  struct prepare_plan final {
    float median;
    EXEC_UNIT void operator()(CostRouteLeg& leg) {
      thrust::get<2>(leg) = thrust::get<1>(leg) <= median;
    }
  };
};

/// Estimates convolutions based on partial plan.
struct estimate_convolutions final {
  size_t size;
  /// Convolution operator.
  struct compare_plan final {
    EXEC_UNIT bool operator()(const thrust::tuple<CostRouteLeg, int>& left,
                              const thrust::tuple<CostRouteLeg, int>& right) const {
      // same vehicle, same plan
      const auto& leftLeg = thrust::get<0>(left);
      const auto& rightLeg = thrust::get<0>(right);
      return thrust::get<0>(leftLeg) == thrust::get<0>(rightLeg) &&
             thrust::get<2>(leftLeg) == thrust::get<2>(rightLeg);
    }
  };

  struct map_leg final {
    EXEC_UNIT thrust::tuple<bool, int> operator()(const thrust::tuple<CostRouteLeg, int>& tuple) {
      return {thrust::get<2>(thrust::get<0>(tuple)), thrust::get<1>(tuple)};
    }
  };

  EXEC_UNIT size_t operator()(const vector_ptr<CostRouteLeg> legs,
                              vector_ptr<thrust::tuple<bool, int>> output,
                              vector_ptr<int> lengths) const {
    auto make_iterator = [&]() {
      return thrust::make_transform_output_iterator(output, map_leg{});
    };
    auto result = thrust::reduce_by_key(
      exec_unit_policy{},
      thrust::make_zip_iterator(thrust::make_tuple(legs, thrust::make_counting_iterator(0))),
      thrust::make_zip_iterator(
        thrust::make_tuple(legs + size, thrust::make_counting_iterator(static_cast<int>(size)))),
      thrust::constant_iterator<int>(1), make_iterator(), lengths, compare_plan{});

    return static_cast<size_t>(thrust::distance(make_iterator(), result.first));
  }
};

/// Creates convolutions based on estimation.
struct create_convolutions final {
  Solution::Shadow solution;
  int convolutionSize;
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

    EXEC_UNIT Convolution operator()(const thrust::tuple<thrust::tuple<bool, int>, int>& tuple) {
      int seq = thrust::get<1>(thrust::get<0>(tuple));
      int length = thrust::get<1>(tuple);

      auto first = base + seq;
      auto last = first + length - 1;

      return vrp::algorithms::common::create_convolution{solution}(base, first, last);
    };
  };

  EXEC_UNIT size_t operator()(int base,
                              const vector_ptr<thrust::tuple<bool, int>> output,
                              const vector_ptr<int> lengths,
                              vector_ptr<Convolution> convolutions) const {
    auto newEnd = thrust::remove_copy_if(
      exec_unit_policy{}, thrust::make_zip_iterator(thrust::make_tuple(output, lengths)),
      thrust::make_zip_iterator(thrust::make_tuple(output + groups, lengths + groups)),
      thrust::make_transform_output_iterator(convolutions, map_group{solution, base}),
      filter_group{convolutionSize});

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

  auto legs = make_unique_ptr_data<CostRouteLeg>(size);
  create_cost_legs{begin, end}.operator()(solution.tasks, *legs);

  auto medians = make_unique_ptr_data<float>(size);
  fill_leg_plan{settings.MedianRatio, static_cast<int>(begin), size}.operator()(*legs, *medians);

  auto output = make_unique_ptr_data<thrust::tuple<bool, int>>(size);
  auto lengths = make_unique_ptr_data<int>(size);
  auto groups = estimate_convolutions{size}.operator()(*legs, *output, *lengths);

  auto convolutions = make_unique_ptr_data<Convolution>(size);
  auto resultSize =
    create_convolutions{solution, settings.ConvolutionSize, size, groups}.operator()(
      static_cast<int>(begin), *output, *lengths, *convolutions);

  return {resultSize, std::move(convolutions)};
}
