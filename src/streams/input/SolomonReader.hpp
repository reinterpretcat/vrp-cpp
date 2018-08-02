#ifndef VRP_STREAMS_SOLOMONREADER_HPP
#define VRP_STREAMS_SOLOMONREADER_HPP

#include "models/Problem.hpp"

#include <functional>

namespace vrp {
namespace streams {

/// Reads classical VRP instances in classical format defined by Solomon.
class SolomonReader final {
public:
  /// Calculates distance between two coordinates.
  using DistanceCalculator =
    std::function<float(const thrust::tuple<double, double>&, const thrust::tuple<double, double>&)>;

  /// Creates VRP problem and resources from input stream.
  vrp::models::Problem read(std::istream& input, const DistanceCalculator& calculator);
};

}  // namespace streams
}  // namespace vrp

#endif  // VRP_STREAMS_SOLOMONREADER_HPP
