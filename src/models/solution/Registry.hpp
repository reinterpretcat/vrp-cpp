#pragma once

#include "models/common/Timestamp.hpp"
#include "models/extensions/solution/Comparators.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Actor.hpp"

#include <gsl/gsl>
#include <range/v3/all.hpp>
#include <set>
#include <unordered_map>
#include <vector>

namespace vrp::models::solution {

/// Specifies an entity responsible for providing actors and keeping track of their usage.
class Registry {
public:
  using SharedActor = std::shared_ptr<const Actor>;

  explicit Registry(const problem::Fleet& fleet) : availableActors_(), allActors_() {
    // TODO we should also consider multiple drivers to support smart vehicle-driver assignment
    Expects(ranges::distance(fleet.drivers()) == 1);
    Expects(ranges::distance(fleet.vehicles()) > 0);

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
          availableActors_[actor->detail].insert(actor);
          allActors_.push_back(actor);
      });
    });

    // clang-format on
  }

  /// Marks actor as used. Returns true whether it is first usage.
  void use(const SharedActor& actor) { availableActors_[actor->detail].erase(actor); }

  /// Marks actor as available.
  void free(const SharedActor& actor) { availableActors_[actor->detail].insert(actor); }

  /// Returns list of all available actors.
  ranges::any_view<const SharedActor> available() const {
    return ranges::view::for_each(availableActors_, [](const auto& pair) { return ranges::view::all(pair.second); });
  }

  /// Returns next possible actors of different types.
  ranges::any_view<const SharedActor> next() const {
    return ranges::view::for_each(
      availableActors_, [](const auto& pair) { return ranges::view::all(pair.second) | ranges::view::take(1); });
  }

  /// Returns all actors.
  ranges::any_view<const SharedActor> all() const { return ranges::view::all(allActors_); }

private:
  /// Specifies available actors grouped by detail.
  std::map<Actor::Detail, std::set<SharedActor>, compare_actor_details> availableActors_;
  std::vector<SharedActor> allActors_;
};
}
