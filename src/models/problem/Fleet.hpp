#pragma once

#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <string>
#include <vector>

namespace vrp::models::problem {

struct Fleet final {
  /// Allow only move.
  Fleet() = default;
  Fleet(Fleet&& other) noexcept : drivers_(std::move(other.drivers_)), vehicles_(std::move(other.vehicles_)) {}
  Fleet(const Fleet&) = delete;
  Fleet& operator=(const Fleet&) = delete;

  Fleet& add(const Driver& driver) {
    drivers_.push_back(std::make_shared<const Driver>(driver));
    return *this;
  }

  Fleet& add(const Vehicle& vehicle) {
    vehicles_.push_back(std::make_shared<const Vehicle>(vehicle));
    return *this;
  }

  ranges::any_view<std::shared_ptr<const Driver>> drivers() const { return ranges::view::all(drivers_); }

  ranges::any_view<std::shared_ptr<const Vehicle>> vehicles() const { return ranges::view::all(vehicles_); }

  auto profiles() const {
    return vehicles() | ranges::view::transform([](const auto& v) { return v->profile; }) |  //
      ranges::to_vector | ranges::action::sort | ranges::action::unique;
  }

private:
  std::vector<std::shared_ptr<const Driver>> drivers_;
  std::vector<std::shared_ptr<const Vehicle>> vehicles_;
};
}