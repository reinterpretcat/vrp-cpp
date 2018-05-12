#include "algorithms/Convolutions.hpp"
#include "utils/Memory.hpp"

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>

using namespace vrp::algorithms;
using namespace vrp::models;
using namespace vrp::utils;
using namespace thrust::placeholders;

using Settings = create_best_convolutions::Settings;

namespace {

/// Contains model shadows.
struct Model final {
  vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;
};

/// Creates cost differences (gradients) as cost change
/// between two customers.
struct create_cost_differences final {

  /// Creates a gradient of costs between tasks.
  struct make_gradient final {
    template<typename Tuple>
    __host__ __device__ Tuple operator()(const Tuple &left,
                                         const Tuple &right) const {
      return thrust::get<0>(left) == thrust::get<0>(right)
             ? thrust::make_tuple(
              thrust::get<0>(left),
              thrust::get<1>(left) - thrust::get<1>(right))
             : left;
    }
  };

  /// Maps cost gradient info to cost.
  struct map_gradient final {
    template <typename Tuple>
    __host__ __device__ float operator()(const Tuple &tuple) const {
      return thrust::get<1>(tuple);
    }
  };

  void operator()(const Tasks &tasks,
                  thrust::device_vector<float> &differences) const {
    thrust::adjacent_difference(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            tasks.vehicles.begin(),
            tasks.costs.begin()
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            tasks.vehicles.end(),
            tasks.costs.end()
        )),
        thrust::make_transform_output_iterator(
            differences.begin(),
            map_gradient()
        ),
        make_gradient()
    );
  }
};

/// Creates partial plan taking into account median of
/// cost differences: customer is served only if cost change
/// is lower than median value.
struct create_partial_plan final {

  void operator()(thrust::device_vector<float> &differences,
                  thrust::device_vector<float> &medians,
                  thrust::device_vector<bool> &plan) const {
    // initialize medians
    thrust::copy(
        thrust::device,
        differences.begin(),
        differences.end(),
        medians.begin()
    );

    // sort to get median
    thrust::sort(
        thrust::device,
        medians.begin(),
        medians.end()
    );
    // TODO
    auto median = medians[medians.size() * 0.5];//MedianRatio];

    // create plan using median
    thrust::transform(
        thrust::device,
        differences.begin(),
        differences.end(),
        plan.begin(),
        _1 <= median
    );
  }
};

/// Estimates convolutions based on partial plan.
struct estimate_convolutions final {
  /// Convolution operator.
  struct compare_plan final {
    template <typename Tuple>
    __host__ __device__ bool operator()(const Tuple &left, const Tuple &right) const {
      return thrust::get<0>(left) == thrust::get<0>(right);
    }
  };

  size_t operator()(const thrust::device_vector<bool> &plan,
                    thrust::device_vector<thrust::tuple<bool, int>> &output,
                    thrust::device_vector<int> &lengths) const {
    return thrust::reduce_by_key(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            plan.begin(),
            thrust::make_counting_iterator(0))
        ),
        thrust::make_zip_iterator(thrust::make_tuple(
            plan.end(),
            thrust::make_counting_iterator(static_cast<int>(plan.size())))
        ),
        thrust::constant_iterator<int>(1),
        output.begin(),
        lengths.begin(),
        compare_plan()
    ).first - output.begin();
  }
};

/// Creates convolutions based on estimation.
struct create_convolutions final {

  /// Filters group by plan and length.
  struct filter_group final {
    template <typename Tuple>
    __host__ __device__ bool operator()(const Tuple &tuple) const {
      // TODO use ConvolutionRatio constant to get proper value
      return !(thrust::get<0>(thrust::get<0>(tuple)) &&
          thrust::get<1>(tuple) > 3);
    }
  };

  /// Maps group to its convolution representation.
  struct map_group final {
    Model *model;
    int base;

    __host__ __device__
    int operator()(int id) const {
      return *(model->problem.customers.demands + id);
    }

    __host__ __device__
    Convolution operator()(const thrust::tuple<thrust::tuple<bool,int>,int> &tuple) {
      auto problem = model->problem;
      auto tasks = model->tasks;

      int seq = thrust::get<1>(thrust::get<0>(tuple));
      int length = thrust::get<1>(tuple);

      auto end = base + seq + 1;
      auto start = end - length;
      auto firstCustomerService = + problem.customers.services[tasks.ids[start]];

      return Convolution {
          // total customers demand
          thrust::transform_reduce(
              thrust::device,
              tasks.ids + start, tasks.ids + end,
              *this,
              0,
              thrust::plus<int>()
          ),
          // new service time from total duration
          tasks.times[end - 1] - tasks.times[start] + firstCustomerService,

          // get fist and last customer
          thrust::make_pair<int,int>(
              tasks.ids[start],
              tasks.ids[end - 1]
          ),
          // get TW which is [first customer TW start, first customer ETA]
          thrust::make_pair<int,int>(
              problem.customers.starts[tasks.ids[start]],
              tasks.times[start] - firstCustomerService
          ),
          // calculate task range (all inclusive)
          thrust::make_pair<int,int>(
              start - base,
              end - base - 1)
      };
    };
  };

  void operator()(const Problem &problem,
                  Tasks &tasks,
                  const thrust::device_vector<thrust::tuple<bool, int>> &output,
                  const thrust::device_vector<int> &lengths,
                  thrust::device_vector<Convolution> &convolutions) const {
    // NOTE there is something weird with transform_output_iterator which
    // forces to allocate shadows explicitly on device when it is used.
    // Otherwise it is crashing even data is not used (problems with copying?..)
    auto model = allocate<Model>({problem.getShadow(), tasks.getShadow()});

    auto newEnd = thrust::remove_copy_if(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            output.begin(),
            lengths.begin())
        ),
        thrust::make_zip_iterator(thrust::make_tuple(
            output.end(),
            lengths.end())
        ),
        thrust::make_transform_output_iterator(
            convolutions.begin(),
            map_group { model.get(), 0 }
        ),
        filter_group()
    );

    release(model);

    // TODO simplify this
    convolutions.resize(static_cast<size_t>(thrust::distance(
        thrust::make_transform_output_iterator(convolutions.begin(), map_group { model.get(), 0 }),
        newEnd))
    );
  }
};

}

create_best_convolutions::Convolutions create_best_convolutions::operator()(const Problem &problem,
                                             Tasks &tasks,
                                             const Settings &settings,
                                             vrp::utils::Pool &pool) {
  auto size = static_cast<size_t>(problem.size());

  auto differences = pool.acquire<thrust::device_vector<float>>(size);
  auto medians = pool.acquire<thrust::device_vector<float>>(size);
  auto plan = pool.acquire<thrust::device_vector<bool>>(size);
  auto output = pool.acquire<thrust::device_vector<thrust::tuple<bool, int>>>(size);
  auto lengths = pool.acquire<thrust::device_vector<int>>(size);

  create_cost_differences{}.operator()(tasks, *differences);

  create_partial_plan{}.operator()(*differences, *medians, *plan);

  size_t groups = estimate_convolutions{}.operator()(*plan, *output, *lengths);

  auto convolutions = pool.acquire<thrust::device_vector<Convolution>>(groups);
  create_convolutions{}.operator()(problem, tasks, *output, *lengths, *convolutions);

  return std::move(convolutions);
}
