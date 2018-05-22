#ifndef VRP_MODELS_LOCATIONS_HPP
#define VRP_MODELS_LOCATIONS_HPP

#include <thrust/tuple.h>
#include <utility>

namespace vrp {
namespace models {

/// Represents host int coordinate.
using HostIntCoord = std::pair<int, int>;
/// Represents host geo coordinate.
using HostGeoCoord = std::pair<double, double>;
/// Represents host int bounding box.
using HostIntBox = std::pair<HostIntCoord, HostIntCoord>;
/// Represents host geo bounding box.
using HostGeoBox = std::pair<HostGeoCoord, HostGeoCoord>;

/// Represents device geo coordinate.
using DeviceGeoCoord = thrust::tuple<double, double>;

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_LOCATIONS_HPP
