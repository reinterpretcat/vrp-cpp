#ifndef VRP_ALGORITHMS_DISTANCES_GEOGRAPHIC_HPP
#define VRP_ALGORITHMS_DISTANCES_GEOGRAPHIC_HPP

#include "models/Locations.hpp"

#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>

namespace vrp {
namespace algorithms {
namespace distances {

/// Calculates geo distance between two coordinates.
template<unsigned int ScaleDown = static_cast<unsigned int>(1E8)>
struct geographic_distance final {
  __host__ __device__ float operator()(const vrp::models::DeviceGeoCoord& left,
                                       const vrp::models::DeviceGeoCoord& right) {
    double leftLon = thrust::get<0>(left) / ScaleDown;
    double leftLat = thrust::get<1>(left) / ScaleDown;
    double rightLon = thrust::get<0>(right) / ScaleDown;
    double rightLat = thrust::get<1>(right) / ScaleDown;

    double dLat = deg2Rad(leftLat - rightLat);
    double dLon = deg2Rad(leftLon - rightLon);

    double lat1 = deg2Rad(leftLat);
    double lat2 = deg2Rad(rightLat);

    double a = std::sin(dLat / 2) * std::sin(dLat / 2) +
               std::sin(dLon / 2) * std::sin(dLon / 2) * std::cos(lat1) * std::cos(lat2);

    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));

    double radius = wgs84EarthRadius(dLat);

    return static_cast<float>(radius * c);
  }

private:
  /// Earth radius at a given latitude, according to the WGS-84 ellipsoid [m].
  __host__ __device__ double wgs84EarthRadius(double lat) {
    // Semi-axes of WGS-84 geoidal reference
    const double WGS84_a = 6378137.0;  // Major semiaxis [m]
    const double WGS84_b = 6356752.3;  // Minor semiaxis [m]

    // http://en.wikipedia.org/wiki/Earth_radius
    auto an = WGS84_a * WGS84_a * std::cos(lat);
    auto bn = WGS84_b * WGS84_b * std::sin(lat);
    auto ad = WGS84_a * std::cos(lat);
    auto bd = WGS84_b * std::sin(lat);
    return std::sqrt((an * an + bn * bn) / (ad * ad + bd * bd));
  }

  /// Converts degrees to radians.
  __host__ __device__ inline double deg2Rad(double degrees) { return M_PI * degrees / 180.0; }
};

}  // namespace distances
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_DISTANCES_GEOGRAPHIC_HPP
