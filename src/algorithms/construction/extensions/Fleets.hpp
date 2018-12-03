#pragma once

#include "models/problem/Fleet.hpp"
#include "models/solution/Actor.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::construction {

/// Returns range of actors without driver and vehicle specified.
inline ranges::any_view<models::solution::Actor>
empty_actors(const models::problem::Fleet& fleet) {
  return ranges::view::for_each(fleet.vehicles(), [](const auto& v) {
    return ranges::yield_from(ranges::view::for_each(ranges::view::all(v->details), [](const auto& d) {
      return ranges::yield(models::solution::Actor{{}, {}, d.start, d.end, d.time});
    }));
  });
}
}