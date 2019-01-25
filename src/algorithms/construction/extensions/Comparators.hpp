#pragma once

#include "models/extensions/solution/Comparators.hpp"

#include <cstdint>

namespace vrp::algorithms::construction {

/// Compares insertion route contexts by their actor.
struct compare_insertion_route_contexts final {
  bool operator()(const InsertionRouteContext& lhs, const InsertionRouteContext& rhs) const {
    auto key1 = reinterpret_cast<std::uintptr_t>(lhs.route->actor.get());
    auto key2 = reinterpret_cast<std::uintptr_t>(rhs.route->actor.get());
    return key1 == key2 ? models::solution::compare_actor_details{}(lhs.route->actor->detail, rhs.route->actor->detail)
                        : key1 < key2;
  }
};
}