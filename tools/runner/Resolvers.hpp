#ifndef VRP_RUNNER_RESOLVERS_HPP
#define VRP_RUNNER_RESOLVERS_HPP


#include <algorithm>
#include <fstream>
#include <thrust/tuple.h>
#include <utility>
#include <vector>

namespace vrp {
namespace utils {

using IntCoord = thrust::tuple<int, int>;
using DoubleCoord = thrust::tuple<double, double>;

using IntBox = thrust::tuple<IntCoord, IntCoord>;
using DoubleBox = thrust::tuple<DoubleCoord, DoubleCoord>;

/// Resolves locations as geo coordinates.
template<typename Mapper>
struct location_resolver final {
  explicit location_resolver(std::fstream& in, const Mapper& mapper) : mapper(mapper) {
    initLocations(in);
    intBoundingBox = getBoundingBox();
  }

  DoubleCoord operator()(int customer) const {
    return mapper(intBoundingBox, locations.at(static_cast<unsigned long>(customer)));
  }

private:
  void initLocations(std::fstream& in) {
    in.clear();
    in.seekg(0, std::ios::beg);

    for (int i = 0; i < 9; ++i)
      in.ignore(std::numeric_limits<std::streamsize>::max(), in.widen('\n'));

    int skip;
    std::pair<int, int> location;
    while (in) {
      in >> skip >> location.first >> location.second >> skip >> skip >> skip >> skip;
      locations.emplace_back(thrust::make_tuple(location.first, location.second));
    }
  }

  IntBox getBoundingBox() const {
    auto minMaxX = std::minmax_element(locations.begin(), locations.end(),
                                       [](const IntCoord& left, const IntCoord& right) {
                                         return thrust::get<0>(left) < thrust::get<0>(right);
                                       });
    auto minMaxY = std::minmax_element(locations.begin(), locations.end(),
                                       [](const IntCoord& left, const IntCoord& right) {
                                         return thrust::get<1>(left) < thrust::get<1>(right);
                                       });

    return thrust::make_tuple(
      IntCoord{thrust::get<0>(*minMaxX.first), thrust::get<1>(*minMaxY.first)},
      IntCoord{thrust::get<0>(*minMaxX.second), thrust::get<1>(*minMaxY.second)});
  };

  const Mapper& mapper;
  std::vector<IntCoord> locations;
  IntBox intBoundingBox;
};


/// Maps int coordinate as double without changes.
struct DefaultMapper final {
  explicit DefaultMapper(double scale) : scale(scale) {}

  DoubleCoord operator()(const IntBox& intBoundingBox, const IntCoord& coordinate) const {
    return thrust::make_tuple(thrust::get<0>(coordinate) * scale,
                              thrust::get<1>(coordinate) * scale);
  }

private:
  double scale;
};

/// Maps int coordinate as geo coordinate inside the bounding box.
struct BoundingBoxMapper final {
  explicit BoundingBoxMapper(const DoubleBox& boundingBox) : geoBoundingBox(boundingBox) {}

  DoubleCoord operator()(const IntBox intBoundingBox, const IntCoord& coordinate) const {
    auto minIntBox = thrust::get<0>(intBoundingBox);
    auto maxIntBox = thrust::get<1>(intBoundingBox);
    auto minGeoBox = thrust::get<0>(geoBoundingBox);
    auto maxGeoBox = thrust::get<1>(geoBoundingBox);

    double ratioX = (thrust::get<0>(coordinate) - thrust::get<0>(minIntBox)) /
                    static_cast<double>(thrust::get<0>(maxIntBox) - thrust::get<0>(minIntBox));

    double ratioY = (thrust::get<1>(coordinate) - thrust::get<1>(minIntBox)) /
                    static_cast<double>(thrust::get<1>(maxIntBox) - thrust::get<1>(minIntBox));

    return thrust::make_tuple(
      thrust::get<0>(minGeoBox) + (thrust::get<0>(maxGeoBox) - thrust::get<0>(minGeoBox)) * ratioX,
      thrust::get<1>(minGeoBox) + (thrust::get<1>(maxGeoBox) - thrust::get<1>(minGeoBox)) * ratioY);
  }

private:
  const DoubleBox geoBoundingBox;
};

}  // namespace utils
}  // namespace vrp

#endif  // VRP_RUNNER_RESOLVERS_HPP
