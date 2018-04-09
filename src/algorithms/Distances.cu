#ifndef VRP_ALGORITHMS_DISTANCES_HPP
#define VRP_ALGORITHMS_DISTANCES_HPP

#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <cmath>

namespace vrp {
namespace algorithms {

/// Calculates cartesian distance between two points on plane in 2D.
struct CartesianDistance {
  __host__ __device__
  float operator()(const thrust::tuple<int, int> &left,
                   const thrust::tuple<int, int> &right) {
    auto x = thrust::get<0>(left) - thrust::get<0>(right);
    auto y = thrust::get<1>(left) - thrust::get<1>(right);
    return static_cast<float>(sqrt(x * x + y * y));
  }
};

/// Calculates geo distance between two coordinates.
struct GeoDistance {
  __host__ __device__
  float operator()(const thrust::tuple<int, int> &left,
                   const thrust::tuple<int, int> &right) {
    auto leftLon = thrust::get<0>(left);
    auto leftLat = thrust::get<1>(left);
    auto rightLon = thrust::get<0>(right);
    auto rightLat = thrust::get<1>(right);

    double dLat = deg2Rad(leftLat - rightLat);
    double dLon = deg2Rad(leftLon - rightLon);

    double lat1 = deg2Rad(leftLat);
    double lat2 = deg2Rad(rightLat);

    double a = std::sin(dLat/2)*std::sin(dLat/2) +
        std::sin(dLon/2)*std::sin(dLon/2)* std::cos(lat1)*std::cos(lat2);

    double c = 2*std::atan2(std::sqrt(a), std::sqrt(1 - a));

    double radius = wgs84EarthRadius(dLat);

    return static_cast<float>(radius*c);
  }
 private:
  /// The circumference at the equator (latitude 0)
  const int LatitudeEquator = 40075160;
  /// Distance of full circle around the earth through the poles.
  const int CircleDistance = 40008000;

  /// Earth radius at a given latitude, according to the WGS-84 ellipsoid [m].
  __host__ __device__
  double wgs84EarthRadius(double lat) {
    // Semi-axes of WGS-84 geoidal reference
    const double WGS84_a = 6378137.0; // Major semiaxis [m]
    const double WGS84_b = 6356752.3; // Minor semiaxis [m]

    // http://en.wikipedia.org/wiki/Earth_radius
    auto an = WGS84_a*WGS84_a*std::cos(lat);
    auto bn = WGS84_b*WGS84_b*std::sin(lat);
    auto ad = WGS84_a*std::cos(lat);
    auto bd = WGS84_b*std::sin(lat);
    return std::sqrt((an*an + bn*bn)/(ad*ad + bd*bd));
  }

  /// converts degrees to radians.
  __host__ __device__
  inline double deg2Rad(double degrees) {
    return M_PI*degrees/180.0;
  }
};


}
}

#endif //VRP_ALGORITHMS_DISTANCES_HPP