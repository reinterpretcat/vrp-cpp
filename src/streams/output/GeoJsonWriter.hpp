#ifndef VRP_STREAMS_GEOJSONWRITER_HPP
#define VRP_STREAMS_GEOJSONWRITER_HPP

#include "models/Tasks.hpp"
#include "models/Locations.hpp"

#include <functional>

namespace vrp {
namespace streams {

/// Writes solution into output stream  in geojson format.
class GeoJsonWriter final {
 public:
  /// Defines resolver func type.
  using LocationResolver = std::function<vrp::models::HostGeoCoord(int)>;

  /// Writes geo json to stream.
  void write(std::ostream &out,
             const vrp::models::Tasks &tasks,
             const LocationResolver &resolver);
};

}
}

#endif //VRP_STREAMS_GEOJSONWRITER_HPP
