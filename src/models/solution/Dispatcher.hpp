#pragma once

#include "models/common/Timestamp.hpp"
#include "models/extensions/solution/Comparators.hpp"
#include "models/problem/Fleet.hpp"
#include "models/solution/Actor.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <set>
#include <unordered_map>

namespace vrp::models::solution {

/// Specifies an entity responsible for dispatching vehicles and drivers.
// TODO at the moment, consider only vehicles, can be extended to support smart vehicle-driver assignment.
struct Dispatcher {
  explicit Dispatcher(const std::shared_ptr<const problem::Fleet>& fleet) : fleet_(fleet) {
    assert(ranges::distance(fleet->drivers()) == 1);
  }

  void use(const Actor& actor) {
    auto result = details_[actor.vehicle->id].insert(actor.detail);

    assert(result.second);
  }

  /// Return available for use actors.
  ranges::any_view<std::shared_ptr<Actor>> actors() const {
    using namespace ranges;
    // TODO this method should also consider different drivers, see comment at top
    // TODO do not return actors wrapped by shared ptr?
    auto driver = *std::begin(fleet_->drivers());

    // clang-format off
    return fleet_->vehicles() | view::for_each([&](const auto& v) {
      auto ds = details_.find(v->id);
      return view::all(v->details) |
          view::transform([&](const auto& d) { return Actor::Detail{d.start, d.end, d.time}; }) |
          view::filter([&](const auto& d) { return ds != details_.end() || ds->second.find(d) != ds->second.end(); }) |
          view::transform([&](const auto& d) { return std::make_shared<Actor>(Actor{v, driver, d}); });
    });
    // clang-format on
  }

private:
  std::shared_ptr<const problem::Fleet> fleet_;
  /// Tracks used vehicles.
  std::unordered_map<std::string, std::set<Actor::Detail, compare_actor_details>> details_;
};
}
