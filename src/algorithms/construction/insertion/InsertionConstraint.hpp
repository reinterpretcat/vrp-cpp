#pragma once

#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "models/solution/Activity.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::algorithms::construction {

/// An insertion constraint which encapsulates behaviour of all possible constraint types.
struct InsertionConstraint final {
  /// Specifies single hard constraint result.
  using HardRouteResult = std::optional<int>;

  /// Specifies hard constraint function which returns empty result or violated constraint code.
  using HardRoute = std::function<HardRouteResult(const InsertionRouteContext& context,
                                                  const ranges::any_view<const models::solution::Activity>&)>;

  /// Specifies soft constraint function which returns additional cost penalty.
  using SoftRoute = std::function<double(const InsertionRouteContext& context)>;

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

  /// Checks whether all hard route constraints are fulfilled.
  /// Returns the code of first failed constraint or empty value.
  HardRouteResult hard(const InsertionRouteContext& ctx,
                       const ranges::any_view<const models::solution::Activity>& view) const {
    return ranges::accumulate(ranges::view::all(hardRouteConstraints_) |
                                ranges::view::transform([&](const auto& constraint) { return constraint(ctx, view); }) |
                                ranges::view::filter([](const auto& result) { return result.has_value(); }) |
                                ranges::view::take(1),
                              HardRouteResult{},
                              [](const auto& acc, const auto& v) { return std::make_optional(v.value()); });
  }

  /// Checks soft route constraints and aggregates associated penalties.
  double soft(const InsertionRouteContext& ctx) const {
    return ranges::accumulate(ranges::view::all(softRouteConstraints_) |
                                ranges::view::transform([&](const auto& constraint) { return constraint(ctx); }),
                              0.0);
  }

  // endregion

private:
  std::vector<HardRoute> hardRouteConstraints_;
  std::vector<SoftRoute> softRouteConstraints_;
};

}  // namespace vrp::algorithms::construction
