#ifndef VRP_UTILS_LOCATIONS_HPP
#define VRP_UTILS_LOCATIONS_HPP

#include <algorithm>
#include <fstream>
#include <utility>

namespace vrp {
namespace utils {

using IntCoord = std::pair<int,int>;
using GeoCoord = std::pair<double,double>;

/// Resolves locations as geo coordinates.
template <typename Mapper>
struct LocationResolver final {
  explicit LocationResolver(std::fstream &in, const Mapper &mapper) : mapper(mapper) {
    initLocations(in);
    intBoundingBox = getBoundingBox();
  }

  GeoCoord operator()(int customer) const {
    return mapper(intBoundingBox, locations.at(static_cast<unsigned long>(customer)));
  }

 private:
  void initLocations(std::fstream &in) {
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

  std::pair<IntCoord,IntCoord> getBoundingBox() const {
    auto minMaxX = std::minmax_element(locations.begin(), locations.end(),
                                       [](const IntCoord &left, const IntCoord &right) {
                                         return left.first < right.first;
                                       });
    auto minMaxY = std::minmax_element(locations.begin(), locations.end(),
                                       [](const IntCoord &left, const IntCoord &right) {
                                         return left.second < right.second;
                                       });

    return std::make_pair(
        IntCoord {minMaxX.first->first, minMaxY.first->second },
        IntCoord {minMaxX.second->first, minMaxY.second->second });
  };

  const Mapper &mapper;
  std::vector<IntCoord> locations;
  std::pair<IntCoord,IntCoord> intBoundingBox;
};

}
}

#endif //VRP_UTILS_LOCATIONS_HPP
