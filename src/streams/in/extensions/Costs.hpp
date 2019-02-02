#pragma once

#include "models/Problem.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"

#include <algorithm>
#include <functional>
#include <range/v3/all.hpp>
#include <tuple>
#include <vector>

namespace vrp::streams::in {

/// Calculates service costs for typical problem.
struct ServiceCosts : models::costs::ActivityCosts {
  models::common::Cost cost(const models::solution::Actor& actor,
                            const models::solution::Activity& activity,
                            const models::common::Timestamp arrival) const override {
    return 0;
  }
};

/// Creates routing matrix as transport costs.
template<typename Distance>
struct RoutingMatrix : models::costs::TransportCosts {
  models::common::Duration duration(const std::string& profile,
                                    const models::common::Location& from,
                                    const models::common::Location& to,
                                    const models::common::Timestamp& departure) const override {
    return distance(profile, from, to, departure);
  }

  models::common::Distance distance(const std::string&,
                                    const models::common::Location& from,
                                    const models::common::Location& to,
                                    const models::common::Timestamp&) const override {
    return matrix_[from * locations_.size() + to];
  }


  auto matrix() const { return ranges::view::all(matrix_); }

  models::common::Location location(int x, int y) {
    // TODO use more performant data structure to have O(1)
    auto location =
      std::find_if(locations_.begin(), locations_.end(), [&](const auto& l) { return l.first == x && l.second == y; });

    if (location != locations_.end())
      return static_cast<models::common::Location>(std::distance(locations_.begin(), location));

    locations_.push_back(std::pair(x, y));
    return locations_.size() - 1;
  }

  void generate() {
    matrix_.reserve(locations_.size() * locations_.size());

    auto distance = Distance{};
    for (size_t i = 0; i < locations_.size(); ++i)
      for (size_t j = 0; j < locations_.size(); ++j) {
        matrix_.push_back(i != j ? distance(locations_[i], locations_[j]) : static_cast<models::common::Distance>(0));
      }
  }

  std::vector<models::common::Distance> matrix_;
  std::vector<std::pair<int, int>> locations_;
};
}
