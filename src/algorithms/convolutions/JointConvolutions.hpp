#ifndef VRP_ALGORITHMS_CONVOLUTIONS_JOINTCONVOLUTIONS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_JOINTCONVOLUTIONS_HPP

#include "algorithms/convolutions/Models.hpp"
#include "models/Solution.hpp"

namespace vrp {
namespace algorithms {
namespace convolutions {

/// Provides the way to create joint pairs of convolutions with
/// additional characteristics.
struct create_joint_convolutions final {
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;
  /// Object pool
  thrust::device_ptr<vrp::utils::DevicePool> pool;

  __device__ JointPairs operator()(const Settings& settings,
                                   const Convolutions& left,
                                   const Convolutions& right) const;
};

}  // namespace convolutions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_CONVOLUTIONS_JOINTCONVOLUTIONS_HPP
