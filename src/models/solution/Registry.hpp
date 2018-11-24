#pragma once

#include "models/common/Timestamp.hpp"
#include "models/extensions/solution/Comparators.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Actor.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <set>
#include <unordered_map>
#include <vector>

namespace vrp::models::solution {

/// Specifies an entity responsible for providing actors and keeping track of their usage.
struct Registry {
  explicit Registry(const std::shared_ptr<const problem::Fleet>& fleet) : actors_(), details_() {
    // TODO we should also consider multiple drivers to support smart vehicle-driver assignment
    assert(ranges::distance(fleet->drivers()) == 1);
    assert(ranges::distance(fleet->vehicles()) > 0);

    using namespace ranges;

    // clang-format off
    actors_ = fleet->vehicles() | view::for_each([&](const auto v) {
      auto drivers = fleet->drivers();
      auto driver = *std::begin(drivers);
      auto vehicle = v;

      return view::all(v->details) |
          view::transform([&](const auto& d) { return Actor::Detail{d.start, d.end, d.time}; }) |
          view::transform([=](const auto& d) { return std::make_shared<const Actor>(Actor{vehicle, driver, d}); });
    });
    // clang-format on
  }

  void use(const Actor& actor) {
    auto result = details_[actor.vehicle->id].insert(actor.detail);

    assert(result.second);
  }

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
