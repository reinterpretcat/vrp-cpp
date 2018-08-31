#include "models/Convolution.hpp"
#include "models/Solution.hpp"
#include "runtime/Config.hpp"

#include <thrust/transform_reduce.h>

namespace vrp {
namespace algorithms {
namespace common {

/// Creates convolution.
struct create_convolution final {
  vrp::models::Solution::Shadow solution;
  ANY_EXEC_UNIT vrp::models::Convolution operator()(int base, int first, int last) {
    auto firstCustomerService = solution.problem.customers.services[solution.tasks.ids[first]];
    auto demand =
      thrust::transform_reduce(vrp::runtime::exec_unit_policy{}, solution.tasks.ids + first,
                               solution.tasks.ids + last + 1, *this, 0, thrust::plus<int>());

    return vrp::models::Convolution{
      base, demand,
      // new service time from total duration
      solution.tasks.times[last] - solution.tasks.times[first] + firstCustomerService,
      // get fist and last customer
      thrust::make_pair<int, int>(solution.tasks.ids[first], solution.tasks.ids[last]),
      // get TW which is [first customer TW start, first customer ETA]
      thrust::make_pair<int, int>(solution.problem.customers.starts[solution.tasks.ids[first]],
                                  solution.tasks.times[first] - firstCustomerService),
      // calculate task range (all inclusive)
      thrust::make_pair<int, int>(first - base, last - base)};
  }

  EXEC_UNIT int operator()(int id) const { return *(solution.problem.customers.demands + id); }
};

}  // namespace common
}  // namespace algorithms
}  // namespace vrp
