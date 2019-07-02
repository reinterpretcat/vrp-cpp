#pragma once

#include <gsl/gsl>
#include <map>
#include <vector>

namespace vrp::streams::in {

/// Represents coordinate index.
struct CoordIndex final {
  void add(const std::vector<double>& location) {
    Expects(location.size() == 2);

    add(location[0], location[1]);
  }

  void add(double latitude, double longitude) {
    auto value = std::make_pair(latitude, longitude);
    if (coordToIndex.find(value) == coordToIndex.end()) {
      auto index = coordToIndex.size();
      coordToIndex[value] = index;
      indexToCoord[index] = value;
    }
  }

  std::vector<double> find(size_t index) const {
    auto pair = indexToCoord.at(index);
    return {pair.first, pair.second};
  }

  size_t find(double latitude, double longitude) const { return coordToIndex.at(std::make_pair(latitude, longitude)); }

  size_t find(const std::vector<double>& location) const {
    Expects(location.size() == 2);

    return find(location[0], location[1]);
  }

private:
  struct coord_less final {
    bool operator()(const std::pair<double, double>& lhs, const std::pair<double, double>& rhs) const {
      return std::tie(lhs.first, lhs.second) < std::tie(rhs.first, rhs.second);
    }
  };

  std::map<std::pair<double, double>, size_t, coord_less> coordToIndex;
  std::map<size_t, std::pair<double, double>> indexToCoord;
};
}