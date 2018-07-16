#ifndef VRP_ALGORITHMS_CONVOLUTIONS_SLICEDCONVOLUTIONS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_SLICEDCONVOLUTIONS_HPP

#include "algorithms/convolutions/Models.hpp"
#include "models/Solution.hpp"

namespace vrp {
namespace algorithms {
namespace convolutions {

/// Returns a "slice" from convolution pairs which can be used
/// to create a new solution.
struct create_sliced_convolutions final {
  /// Solution shadow.
  vrp::models::Solution::Shadow solution;

  EXEC_UNIT Convolutions operator()(const Settings& settings, const JointPairs& pairs) const;
};

}  // namespace convolutions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_CONVOLUTIONS_SLICEDCONVOLUTIONS_HPP
