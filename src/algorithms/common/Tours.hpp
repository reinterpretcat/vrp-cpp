#ifndef VRP_ALGORITHMS_COMMON_TOURS_HPP
#define VRP_ALGORITHMS_COMMON_TOURS_HPP

#include "runtime/Config.hpp"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>

namespace vrp {
namespace algorithms {
namespace common {

/// Finds tours in solution.
template<typename VehicleIterator, typename OutputIterator>
struct find_tours final {
  /// Specifies vehicle range (tour): start, end, id, and some extra.
  using VehicleRange = thrust::tuple<int, int, int, int>;

  /// Represents operator which helps to create vehicle ranges without extra memory footprint.
  struct create_vehicle_ranges final {
    EXEC_UNIT VehicleRange operator()(const VehicleRange& left, const VehicleRange& right) {
      auto leftStart = thrust::get<0>(left);
      auto leftEnd = thrust::get<1>(left);
      auto leftVehicle = thrust::get<2>(left);
      auto leftExtra = thrust::get<3>(left);

      auto rightStart = thrust::get<0>(right);
      auto rightEnd = thrust::get<1>(right);
      auto rightVehicle = thrust::get<2>(right);

      if (rightStart == 0) return {1, leftExtra != -1 ? 1 : leftEnd, 0, -1};

      if (leftExtra != -1) {
        // continue with this vehicle
        if (leftExtra == rightVehicle) {
          return {-1, leftStart - 1, leftExtra, -1};
        }
        // vehicle was used only once
        else {
          return {leftStart - 1, leftStart - 1, leftExtra, rightVehicle};
        }
      }

      if (leftVehicle != rightVehicle) return {rightStart + 1, leftEnd, leftVehicle, rightVehicle};

      return {-1, leftEnd, leftVehicle, -1};
    }
  };

  int last;

  EXEC_UNIT void operator()(VehicleIterator begin, VehicleIterator end, OutputIterator out) {
    int size = static_cast<int>(thrust::distance(begin, end));

    thrust::exclusive_scan(
      vrp::runtime::exec_unit_policy{},

      thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_reverse_iterator(thrust::make_counting_iterator(size)),
                           thrust::make_constant_iterator(-1), thrust::make_reverse_iterator(end),
                           thrust::make_constant_iterator(-1))),

      thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_reverse_iterator(thrust::make_counting_iterator(-1)),
                           thrust::make_constant_iterator(1), thrust::make_reverse_iterator(begin),
                           thrust::make_constant_iterator(-1))),

      out,

      VehicleRange{-1, size - 1, last, -1}, create_vehicle_ranges{});
  }
};

}  // namespace common
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_COMMON_TOURS_HPP