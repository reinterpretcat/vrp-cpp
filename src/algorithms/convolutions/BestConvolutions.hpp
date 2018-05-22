#ifndef VRP_ALGORITHMS_CONVOLUTIONS_BESTCONVOLUTIONS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_BESTCONVOLUTIONS_HPP

#include "algorithms/convolutions/Models.hpp"

namespace vrp {
namespace algorithms {
namespace convolutions {

/// Creates a group of customers (convolution) which can be served virtually as one.
struct create_best_convolutions final {
  vrp::models::Convolutions operator()(const vrp::models::Problem& problem,
                                       vrp::models::Tasks& tasks,
                                       const Settings& settings) const;
};

}  // namespace convolutions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_CONVOLUTIONS_BESTCONVOLUTIONS_HPP
