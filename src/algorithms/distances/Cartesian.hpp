#ifndef VRP_ALGORITHMS_DISTANCES_CARTESIAN_HPP
#define VRP_ALGORITHMS_DISTANCES_CARTESIAN_HPP

#include "runtime/Config.hpp"

#include <cmath>
#include <thrust/tuple.h>

namespace vrp {
namespace algorithms {
namespace distances {

/// Calculates cartesian distance between two points on plane in 2D.
struct cartesian_distance final {
  ANY_EXEC_UNIT float operator()(const thrust::tuple<double, double>& left,
                                 const thrust::tuple<double, double>& right) {
    auto x = thrust::get<0>(left) - thrust::get<0>(right);
    auto y = thrust::get<1>(left) - thrust::get<1>(right);
    return static_cast<float>(sqrt(x * x + y * y));
  }
};

}  // namespace distances
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_DISTANCES_CARTESIAN_HPP
