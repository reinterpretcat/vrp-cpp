#pragma once

#include "models/common/Location.hpp"
#include "models/problem/Actor.hpp"

namespace vrp::models::problem {

/// Provides the way to get routing information.
struct Routing {
  /// Returns transport time between two locations.
  virtual std::uint64_t time(const common::Location& from,
                             const common::Location& to,
                             const problem::Actor& actor,
                             std::uint64_t departure) = 0;

  /// Returns transport cost between two locations.
  virtual std::uint64_t cost(const common::Location& from,
                             const common::Location& to,
                             const problem::Actor& actor,
                             std::uint64_t departure) = 0;
};

}  // namespace vrp::models::problem