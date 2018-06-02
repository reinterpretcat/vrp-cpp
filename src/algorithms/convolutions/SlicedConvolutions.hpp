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
  vrp::models::Convolutions operator()(vrp::models::Solution& solution,
                                       const Settings& settings,
                                       const JointPairs& pairs) const;
};

}  // namespace convolutions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_CONVOLUTIONS_SLICEDCONVOLUTIONS_HPP
