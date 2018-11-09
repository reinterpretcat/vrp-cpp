#pragma once

#include "models/common/Location.hpp"
#include "models/common/Timestamp.hpp"
#include "models/problem/Actor.hpp"

namespace vrp::models::problem {

/// Provides the way to get routing information for specific locations.
struct Routing {
  /// Returns transport time between two locations.
  virtual std::uint64_t time(const problem::Actor& actor,
                             const common::Location& from,
                             const common::Location& to,
                             const common::Timestamp& departure) = 0;

  /// Returns transport cost between two locations.
  virtual std::uint64_t cost(const problem::Actor& actor,
                             const common::Location& from,
                             const common::Location& to,
                             const common::Timestamp& departure) = 0;
};

}  // namespace vrp::models::problem