#pragma once

#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <string>
#include <unordered_map>

namespace vrp::models::problem {

struct Fleet final {
  /// Allow only move.
  Fleet() = default;
  Fleet(Fleet&& other) : drivers_(std::move(other.drivers_)), vehicles_(std::move(other.vehicles_)) {}
  Fleet(const Fleet&) = delete;
  Fleet& operator=(const Fleet&) = delete;

  Fleet& add(const Driver& driver) {
    if (drivers_.find(driver.id) != drivers_.end())
      throw std::invalid_argument("Driver is already added to the fleet.");
    drivers_.insert({driver.id, std::make_shared<const Driver>(driver)});
    return *this;
  }

  Fleet& add(const Vehicle& vehicle) {
    if (vehicles_.find(vehicle.id) != vehicles_.end())
      throw std::invalid_argument("Vehicle is already added to the fleet.");
    vehicles_.insert({vehicle.id, std::make_shared<const Vehicle>(vehicle)});
    return *this;
  }

  ranges::any_view<std::shared_ptr<const Driver>> drivers() const {
    return ranges::view::all(drivers_) | ranges::view::transform([](const auto& d) { return d.second; });
  }

  std::shared_ptr<const Driver> driver(const std::string& id) const {
    auto result = drivers_.find(id);
    if (result == drivers_.end()) throw std::invalid_argument(std::string("Cannot find driver with id:") + id);
    return result->second;
  }

  ranges::any_view<std::shared_ptr<const Vehicle>> vehicles() const {
    return ranges::view::all(vehicles_) | ranges::view::transform([](const auto& v) { return v.second; });
  }

  std::shared_ptr<const Vehicle> vehicle(const std::string& id) const {
    auto result = vehicles_.find(id);
    if (result == vehicles_.end()) throw std::invalid_argument(std::string("Cannot find vehicle with id:") + id);
    return result->second;
  }

  auto profiles() const {
    return vehicles() | ranges::view::transform([](const auto& v) { return v->profile; }) |  //
      ranges::to_vector | ranges::action::sort | ranges::action::unique;
  }

private:
  std::unordered_map<std::string, std::shared_ptr<const Driver>> drivers_;
  std::unordered_map<std::string, std::shared_ptr<const Vehicle>> vehicles_;
};
}