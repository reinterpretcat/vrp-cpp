#pragma once

#include "algorithms/constraints/HardRouteConstraint.hpp"
#include "algorithms/constraints/SoftRouteConstraint.hpp"

#include <memory>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::algorithms::constraints {

/// A composite constraint which encapsulates behaviour of all possible constraint types.
struct CompositeConstraint final : public HardRouteConstraint, public SoftRouteConstraint {

  // region Add

  /// Adds hard route constraints.
  CompositeConstraint& add(const std::shared_ptr<HardRouteConstraint>& constraint) {
    hardRouteConstraints_.push_back(constraint);
    return *this;
  }

  /// Adds soft route constraints.
  CompositeConstraint& add(const std::shared_ptr<SoftRouteConstraint>& constraint) {
    softRouteConstraints_.push_back(constraint);
    return *this;
  }

  // endregion

  // region Implementation

  bool fulfilled() const override {
    return ranges::accumulate(
      ranges::view::all(hardRouteConstraints_) | ranges::view::transform([](const auto& c) { return c->fulfilled(); }),
      true, [](const auto& lhs, const auto& rhs) { return lhs && rhs; });
  }

  double cost() const override {
    // TODO
    return 0;
  }

  // endregion

private:
  std::vector<std::shared_ptr<HardRouteConstraint>> hardRouteConstraints_;
  std::vector<std::shared_ptr<SoftRouteConstraint>> softRouteConstraints_;
};

}  // namespace vrp::algorithms::constraints
