#pragma once

#include "algorithms/construction/insertion/InsertionContext.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::algorithms::construction {

/// An insertion constraint which encapsulates behaviour of all possible constraint types.
struct InsertionConstraint final {
  /// Specifies single hard constraint result.
  using HardResult = std::optional<int>;

  /// Specifies hard constraint function which returns empty result or violated constraint code.
  using HardRoute = std::function<HardResult(const InsertionContext& context)>;

  /// Specifies soft constraint function which returns additional cost penalty.
  using SoftRoute = std::function<double()>;

  // region Add

  /// Adds hard route constraints.
  InsertionConstraint& add(HardRoute constraint) {
    hardRouteConstraints_.push_back(constraint);
    return *this;
  }

  /// Adds soft route constraints.
  InsertionConstraint& add(SoftRoute constraint) {
    softRouteConstraints_.push_back(constraint);
    return *this;
  }

  // endregion

  // region Implementation

  HardResult hard(const InsertionContext& ctx) const {
    return ranges::accumulate(ranges::view::all(hardRouteConstraints_) |
                                ranges::view::transform([&ctx](const auto& constraint) { return constraint(ctx); }) |
                                ranges::view::filter([](const auto& result) { return result.has_value(); }) |
                                ranges::view::take(1),
                              HardResult{},
                              [](const auto& acc, const auto& v) { return std::make_optional(v.value()); });
  }

  double soft() const {
    // TODO
    return 0;
  }

  // endregion

private:
  std::vector<HardRoute> hardRouteConstraints_;
  std::vector<SoftRoute> softRouteConstraints_;
};

}  // namespace vrp::algorithms::construction
