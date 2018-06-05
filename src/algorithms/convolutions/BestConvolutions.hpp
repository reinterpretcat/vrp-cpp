#ifndef VRP_ALGORITHMS_CONVOLUTIONS_BESTCONVOLUTIONS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_BESTCONVOLUTIONS_HPP

#include "algorithms/convolutions/Models.hpp"
#include "models/Solution.hpp"

namespace vrp {
namespace algorithms {
namespace convolutions {

/// Creates a group of customers (convolution) which can be served virtually as one.
struct create_best_convolutions final {
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;
  /// Object pool
  thrust::device_ptr<vrp::utils::DevicePool> pool;

  __device__ Convolutions operator()(const Settings& settings, int index) const;
};

}  // namespace convolutions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_CONVOLUTIONS_BESTCONVOLUTIONS_HPP
