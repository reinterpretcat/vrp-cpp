#pragma once

#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <set>
#include <string>
#include <vector>

namespace vrp::models::problem {

struct Fleet final {
  /// Allow only move.
  Fleet() = default;
  Fleet(Fleet&& other) noexcept :
    drivers_(std::move(other.drivers_)),
    vehicles_(std::move(other.vehicles_)),
    profiles_(other.profiles_) {}
  Fleet(const Fleet&) = delete;
  Fleet& operator=(const Fleet&) = delete;

  Fleet& add(const Driver& driver) {
    drivers_.push_back(std::make_shared<const Driver>(driver));
    return *this;
  }

  Fleet& add(const Vehicle& vehicle) {
    vehicles_.push_back(std::make_shared<const Vehicle>(vehicle));
    profiles_.insert(vehicle.profile);

    return *this;
  }

  ranges::any_view<std::shared_ptr<const Driver>> drivers() const { return ranges::view::all(drivers_); }

  ranges::any_view<std::shared_ptr<const Vehicle>> vehicles() const { return ranges::view::all(vehicles_); }

  ranges::any_view<std::string> profiles() const { return ranges::view::all(profiles_); }

private:
  std::vector<std::shared_ptr<const Driver>> drivers_;
  std::vector<std::shared_ptr<const Vehicle>> vehicles_;
  std::set<std::string> profiles_;
};
}