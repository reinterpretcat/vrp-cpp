#pragma once

#include "models/problem/Job.hpp"
#include "models/solution/Actor.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace vrp::models {

/// Specifies jobs locked to specific actors.
struct Lock final {
  /// Specifies locked jobs type.
  using Jobs = std::vector<problem::Job>;

  /// Specifies condition when locked jobs can be assigned to specific actor.
  using Condition = std::function<bool(const solution::Actor&)>;

  /// Specifies how jobs should be ordered in tour.
  enum class Order {
    /// Jobs can be reshuffled in any order.
    Any,
    /// Jobs cannot be reshuffled, but new job can be inserted in between.
    Sequence,
    /// Jobs cannot be reshuffled and no jobs can be inserted in between.
    Strict
  };

  /// Specifies how other jobs can be inserted in tour.
  struct Position {
    bool stickToDeparture;
    bool stickToArrival;

    static Position middle() { return Position{false, false}; }
    static Position departure() { return Position{true, false}; }
    static Position arrival() { return Position{false, true}; }
    static Position fixed() { return Position{true, true}; }
  };

  /// Specifies multiple details.
  struct Detail {
    /// Order type.
    Order order;

    /// Insertion type.
    Position position;

    /// Locked jobs.
    Jobs jobs;
  };

  /// Filter condition
  Condition condition;

  /// Details.
  std::vector<Detail> details;
};
}