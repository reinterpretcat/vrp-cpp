#ifndef VRP_ALGORITHMS_DISTANCES_HPP
#define VRP_ALGORITHMS_DISTANCES_HPP

#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <cmath>

namespace vrp {
namespace algorithms {

/// Calculates cartesian distance between two points on plane in 2D.
struct CartesianDistances {
  __host__ __device__
  float operator()(const thrust::tuple<int, int> &left,
                   const thrust::tuple<int, int> &right) {
    auto x = thrust::get<0>(left) - thrust::get<0>(right);
    auto y = thrust::get<1>(left) - thrust::get<1>(right);
    return static_cast<float>(sqrt(x * x + y * y));
  }
};

}
}

#endif //VRP_ALGORITHMS_DISTANCES_HPP