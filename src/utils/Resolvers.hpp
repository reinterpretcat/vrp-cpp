#ifndef VRP_UTILS_RESOLVERS_HPP
#define VRP_UTILS_RESOLVERS_HPP

#include "models/Locations.hpp"

#include <algorithm>
#include <fstream>
#include <utility>
#include <vector>

namespace vrp {
namespace utils {

/// Resolves locations as geo coordinates.
template<typename Mapper>
struct location_resolver final {
  explicit location_resolver(std::fstream& in, const Mapper& mapper) : mapper(mapper) {
    initLocations(in);
    intBoundingBox = getBoundingBox();
  }

  vrp::models::HostGeoCoord operator()(int customer) const {
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
      locations.emplace_back(location);
    }
  }

  vrp::models::HostIntBox getBoundingBox() const {
    auto minMaxX = std::minmax_element(
      locations.begin(), locations.end(),
      [](const vrp::models::HostIntCoord& left, const vrp::models::HostIntCoord& right) {
        return left.first < right.first;
      });
    auto minMaxY = std::minmax_element(
      locations.begin(), locations.end(),
      [](const vrp::models::HostIntCoord& left, const vrp::models::HostIntCoord& right) {
        return left.second < right.second;
      });

    return std::make_pair(vrp::models::HostIntCoord{minMaxX.first->first, minMaxY.first->second},
                          vrp::models::HostIntCoord{minMaxX.second->first, minMaxY.second->second});
  };

  const Mapper& mapper;
  std::vector<vrp::models::HostIntCoord> locations;
  vrp::models::HostIntBox intBoundingBox;
};

}  // namespace utils
}  // namespace vrp

#endif  // VRP_UTILS_RESOLVERS_HPP
