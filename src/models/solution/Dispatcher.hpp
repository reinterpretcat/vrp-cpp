#pragma once

#include "models/problem/Fleet.hpp"
#include "models/solution/Actor.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <unordered_map>

namespace vrp::models::solution {

/// Specifies an entity responsible for dispatching vehicles and drivers.
struct Dispatcher {
  explicit Dispatcher(const std::shared_ptr<const problem::Fleet>& fleet) : fleet_(fleet) {}

  void use(const Actor& actor) {
    // assert ()
  }

  /// Return available for use actors.
  auto actors() {}


private:
  problem::VehicleDetail asDetail(const Actor& actor) const {
    return problem::VehicleDetail{actor.start, actor.end, actor.time};
  }

  std::shared_ptr<const problem::Fleet> fleet_;

  std::unordered_map<std::string, problem::VehicleDetail> details_;
};
}
