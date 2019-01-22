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
class Registry {
public:
  using SharedActor = std::shared_ptr<const Actor>;

  explicit Registry(const problem::Fleet& fleet) : actors_() {
    // TODO we should also consider multiple drivers to support smart vehicle-driver assignment
    assert(ranges::distance(fleet.drivers()) == 1);
    assert(ranges::distance(fleet.vehicles()) > 0);

    using namespace ranges;

    // TODO consider support for more than one driver
    auto drivers = fleet.drivers();
    auto driver = *std::begin(drivers);

    // clang-format off

    ranges::for_each(fleet.vehicles(), [&](const auto& vehicle) {
      ranges::for_each(vehicle->details | view::transform([&](const auto& d) {
          return std::make_shared<const Actor>(Actor{vehicle, driver, Actor::Detail{d.start, d.end, d.time}});
        }),
        [&](const auto& actor) {
          actors_[actor->detail].insert(actor);
      });
    });

    // clang-format on
  }

  /// Marks actor as used. Returns true whether it is first usage.
  void use(const SharedActor& actor) { actors_[actor->detail].erase(actor); }

  /// Marks actor as available.
  void free(const SharedActor& actor) { actors_[actor->detail].insert(actor); }

  /// Returns next possible actors of different types.
  ranges::any_view<const SharedActor> next() const {
    return ranges::view::for_each(
      actors_, [](const auto& pair) { return ranges::view::all(pair.second) | ranges::view::take(1); });
  }

private:
  /// Specifies available actors grouped by detail.
  std::map<Actor::Detail, std::set<SharedActor>, compare_actor_details> actors_;
};
}
