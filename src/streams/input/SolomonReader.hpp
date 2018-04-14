#ifndef VRP_STREAMS_SOLOMONREADER_HPP
#define VRP_STREAMS_SOLOMONREADER_HPP

#include "models/Problem.hpp"
#include "models/Locations.hpp"

#include <functional>

namespace vrp {
namespace streams {

/// Reads classical VRP instances in classical format defined by Solomon.
class SolomonReader final {
public:
  /// Calculates distance between two coordinates.
  using DistanceCalculator = std::function<float(const vrp::models::DeviceGeoCoord&,
                                                 const vrp::models::DeviceGeoCoord&)>;

  /// Creates VRP problem and resources from input stream.
  vrp::models::Problem read(std::istream &input,
                            const DistanceCalculator& calculator);
};

}
}

#endif //VRP_STREAMS_SOLOMONREADER_HPP