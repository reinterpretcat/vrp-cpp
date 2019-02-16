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
  using DurationProfiles = std::map<std::string, std::vector<common::Duration>>;
  using DistanceProfiles = std::map<std::string, std::vector<common::Distance>>;

  MatrixTransportCosts(DurationProfiles&& durations, DistanceProfiles&& distances) :
    durations_(std::move(durations)),
    distances_(std::move(distances)),
    size_(0) {
    assert(durations_.size() == distances_.size());
    ranges::for_each(durations, [this](const auto& pair) mutable {
      if (this->size_ == 0) {
        this->size_ = std::sqrt(pair.second.size());
        assert(this->size_ * this->size_ == pair.second.size());
      }
      assert(this->size_ == std::sqrt(pair.second.size()));
      assert(this->size_ == std::sqrt(this->distances_.at(pair.first).size()));
    });
  }

  MatrixTransportCosts(MatrixTransportCosts&& other) :
    durations_(std::move(other.durations_)),
    distances_(std::move(other.distances_)),
    size_(other.size_) {}
  MatrixTransportCosts(const MatrixTransportCosts&) = delete;
  MatrixTransportCosts& operator=(const MatrixTransportCosts&) = delete;

  /// Returns transport time between two locations.
  common::Duration duration(const std::string& profile,
                            const common::Location& from,
                            const common::Location& to,
                            const common::Timestamp& departure) const override {
    return durations_.at(profile).at(from * size_ + to);
  }

  /// Returns transport distance between two locations.
  common::Distance distance(const std::string& profile,
                            const common::Location& from,
                            const common::Location& to,
                            const common::Timestamp& departure) const override {
    return distances_.at(profile).at(from * size_ + to);
  }

private:
  const DurationProfiles durations_;
  const DistanceProfiles distances_;
  int size_;
};
}