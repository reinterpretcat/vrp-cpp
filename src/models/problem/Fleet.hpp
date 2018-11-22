#pragma once

#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <string>
#include <unordered_map>

namespace vrp::models::problem {

struct Fleet final {
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

  auto drivers() const {
    return ranges::view::all(drivers_) | ranges::view::transform([](const auto& d) { return d.second; });
  }

  auto vehicles() const {
    return ranges::view::all(vehicles_) | ranges::view::transform([](const auto& v) { return v.second; });
  }

  std::shared_ptr<const Vehicle> vehicle(const std::string& id) const {
    auto result = vehicles_.find(id);
    if (result == vehicles_.end()) throw std::invalid_argument(std::string("Cannot find vehicle with id:") + id);
    return result->second;
  }

private:
  std::unordered_map<std::string, std::shared_ptr<const Driver>> drivers_;
  std::unordered_map<std::string, std::shared_ptr<const Vehicle>> vehicles_;
};
}