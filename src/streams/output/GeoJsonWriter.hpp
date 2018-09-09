#ifndef VRP_STREAMS_GEOJSONWRITER_HPP
#define VRP_STREAMS_GEOJSONWRITER_HPP

#include "models/Tasks.hpp"

#include <functional>

namespace vrp {
namespace streams {

/// Writes solution into output stream  in geojson format.
class GeoJsonWriter final {
public:
  /// Defines resolver func type.
  using LocationResolver = std::function<thrust::tuple<double, double>(int)>;

  /// Writes geo json to stream.
  void write(std::ostream& out, const vrp::models::Tasks& tasks, const LocationResolver& resolver);

  /// Writes geo json to stream.
  void write(std::ostream& out, const vrp::models::Tasks& tasks, const LocationResolver& resolver, int solutionId);
};

}  // namespace streams
}  // namespace vrp

#endif  // VRP_STREAMS_GEOJSONWRITER_HPP
