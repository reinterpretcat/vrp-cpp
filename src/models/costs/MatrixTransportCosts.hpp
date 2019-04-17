#pragma once

#include "models/costs/TransportCosts.hpp"

#include <map>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::models::costs {

/// Uses custom distance and duration matrices as source of transport cost information.
/// Not time aware as it ignores departure timestamp.
class MatrixTransportCosts : public TransportCosts {
public:
  using DurationProfiles = std::vector<std::vector<common::Duration>>;
  using DistanceProfiles = std::vector<std::vector<common::Distance>>;

  MatrixTransportCosts(DurationProfiles&& durations, DistanceProfiles&& distances) :
    durations_(std::move(durations)),
    distances_(std::move(distances)),
    size_(0) {
    assert(!durations_.empty());
    assert(durations_.size() == distances_.size());

    auto size = std::sqrt(durations_.front().size());
    assert(ranges::all_of(durations_, [size](const auto& d) { return std::sqrt(d.size()) == size; }));
    assert(ranges::all_of(distances_, [size](const auto& d) { return std::sqrt(d.size()) == size; }));

    size_ = size;
  }

  MatrixTransportCosts(MatrixTransportCosts&& other) :
    durations_(std::move(other.durations_)),
    distances_(std::move(other.distances_)),
    size_(other.size_) {}
  MatrixTransportCosts(const MatrixTransportCosts&) = delete;
  MatrixTransportCosts& operator=(const MatrixTransportCosts&) = delete;

  /// Returns transport time between two locations.
  common::Duration duration(const common::Profile profile,
                            const common::Location& from,
                            const common::Location& to,
                            const common::Timestamp& departure) const override {
    return durations_[profile][from * size_ + to];
  }

  /// Returns transport distance between two locations.
  common::Distance distance(const common::Profile profile,
                            const common::Location& from,
                            const common::Location& to,
                            const common::Timestamp& departure) const override {
    return distances_[profile][from * size_ + to];
  }

private:
  const DurationProfiles durations_;
  const DistanceProfiles distances_;
  int size_;
};
}