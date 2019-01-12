#pragma once

#include "models/common/Timestamp.hpp"
#include "models/extensions/solution/Comparators.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Actor.hpp"

#include <range/v3/all.hpp>
#include <set>
#include <unordered_map>
#include <vector>

namespace vrp::models::solution {

/// Specifies an entity responsible for providing actors and keeping track of their usage.
struct Registry {
  explicit Registry(const problem::Fleet& fleet) : actors_(), details_() {
    // TODO we should also consider multiple drivers to support smart vehicle-driver assignment
    assert(ranges::distance(fleet.drivers()) == 1);
    assert(ranges::distance(fleet.vehicles()) > 0);

    using namespace ranges;

    // clang-format off
    actors_ = fleet.vehicles() | view::for_each([&](const auto v) {
      auto drivers = fleet.drivers();
      auto driver = *std::begin(drivers);
      auto vehicle = v;

      return view::all(v->details) |
          view::transform([&](const auto& d) { return Actor::Detail{d.start, d.end, d.time}; }) |
          view::transform([=](const auto& d) { return std::make_shared<const Actor>(Actor{vehicle, driver, d}); });
    });
    // clang-format on
  }

  /// Marks actor as used. Returns true whether it is first usage.
  bool use(const Actor& actor) { return details_[actor.vehicle->id].insert(actor.detail).second; }

  /// Marks actor as available.
  void free(const Actor& actor) { details_[actor.vehicle->id].erase(actor.detail); }

  /// Return available for use actors.
  ranges::any_view<std::shared_ptr<const Actor>> actors() const {
    return ranges::view::all(actors_) | ranges::view::remove_if([&](const auto& a) {
             auto set = details_.find(a->vehicle->id);
             auto ctx = std::pair(set, (set != details_.end()));
             return ctx.second && ctx.first->second.find(a->detail) != ctx.first->second.end();
           });
  }

private:
  std::vector<std::shared_ptr<const Actor>> actors_;
  std::unordered_map<std::string, std::set<Actor::Detail, compare_actor_details>> details_;
};
}
