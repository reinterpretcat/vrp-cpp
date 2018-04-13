#include "algorithms/Distances.cu"
#include "streams/input/SolomonReader.cu"
#include "streams/output/GeoJsonWriter.hpp"
#include "heuristics/NearestNeighbor.hpp"
#include "solver/genetic/Populations.hpp"
#include "utils/Locations.hpp"

using namespace vrp::algorithms;
using namespace vrp::heuristics;
using namespace vrp::genetic;
using namespace vrp::streams;
using namespace vrp::utils;

namespace {
/// Maps int coordinate as double without changes.
struct DefaultMapper final {
  explicit DefaultMapper(double scale) : scale(scale) {}

  GeoCoord operator()(const std::pair<IntCoord,IntCoord> intBoundingBox,
                      const GeoCoord &coordinate) const {
    return std::make_pair(coordinate.first * scale, coordinate.second * scale);
  }
 private:
  double scale;
};

/// Maps int coordinate as geo coordinate inside the bounding box.
struct BoundingBoxMapper final {
  explicit BoundingBoxMapper(const std::pair<GeoCoord,GeoCoord> &boundingBox) :
    geoBoundingBox(boundingBox) {}

  GeoCoord operator()(const std::pair<IntCoord,IntCoord> intBoundingBox,
                      const GeoCoord &coordinate) const {

    double ratioX = (coordinate.first - intBoundingBox.first.first) /
        static_cast<double>(intBoundingBox.second.first - intBoundingBox.first.first);

    double ratioY = (coordinate.second- intBoundingBox.first.second) /
        static_cast<double>(intBoundingBox.second.second - intBoundingBox.first.second);

    return std::make_pair(
        geoBoundingBox.first.first + (geoBoundingBox.second.first - geoBoundingBox.first.first) * ratioX,
        geoBoundingBox.first.second + (geoBoundingBox.second.second - geoBoundingBox.first.second) * ratioY);
  }
 private:
  const std::pair<GeoCoord,GeoCoord> geoBoundingBox;
};

}

int main(int argc, char* argv[]) {
  if (argc != 3)
    throw std::invalid_argument("Missing input or output argument.");

  std::fstream in(argv[1], std::fstream::in);
  std::fstream out(argv[2], std::fstream::out);

  auto problem = SolomonReader<geographic_distance<>>().read(in);
  auto solution = create_population<NearestNeighbor>(problem)({ 1 });

  auto mapper = BoundingBoxMapper({ {13.3285, 52.4915}, {13.4663, 52.5553} });
  auto resolver = LocationResolver<decltype(mapper)>(in, mapper);
  GeoJsonWriter<decltype(resolver)>().write(out, solution, resolver);

  return 0;
}
