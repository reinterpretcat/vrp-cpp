#pragma once

#include "algorithms/construction/insertion/InsertionActivityContext.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/solution/Activity.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::algorithms::construction {

/// An insertion constraint which encapsulates behaviour of all possible constraint types.
struct InsertionConstraint final {
  /// Specifies activities collection.
  using Activities = ranges::any_view<const models::solution::Activity>;

  using Cost = models::common::Cost;

  // region Route types

  /// Specifies single hard constraint result.
  using HardRouteResult = std::optional<int>;

  /// Specifies hard route constraint function which returns empty result or violated constraint code.
  using HardRoute = std::function<HardRouteResult(const InsertionRouteContext&, const Activities&)>;

  /// Specifies soft route constraint function which returns additional cost penalty.
  using SoftRoute = std::function<Cost(const InsertionRouteContext& context, const Activities&)>;

  // endregion

  // Activity types

  /// Specifies hard activity constraint function which returns empty result or violated constraint code.
  using HardActivity = std::function<ConstraintStatus(const InsertionRouteContext&, const InsertionActivityContext&)>;

  /// Specifies soft activity constraint function which returns additional cost penalty.
  using SoftActivity = std::function<Cost(const InsertionRouteContext&, const InsertionActivityContext&)>;

  // endregion

  // region Add route

  /// Adds hard route constraints
  InsertionConstraint& addHardRoute(HardRoute constraint) {
    hardRouteConstraints_.push_back(constraint);
    return *this;
  }

  /// Adds soft route constraint.
  InsertionConstraint& addSoftRoute(SoftRoute constraint) {
    softRouteConstraints_.push_back(constraint);
    return *this;
  }

  // endregion

  // region Add activity

  /// Adds hard activity constraint.
  InsertionConstraint& addHardActivity(HardActivity constraint) {
    hardActivityConstraints_.push_back(constraint);
    return *this;
  }

  /// Adds soft activity constraint.
  InsertionConstraint& addSoftActivity(SoftActivity constraint) {
    softActivityConstraints_.push_back(constraint);
    return *this;
  }

  // endregion

  // region Route evaluations

  /// Checks whether all hard route constraints are fulfilled.
  /// Returns the code of first failed constraint or empty value.
  HardRouteResult hard(const InsertionRouteContext& ctx, const Activities& acts) const {
    return ranges::accumulate(ranges::view::all(hardRouteConstraints_) |
                                ranges::view::transform([&](const auto& constraint) { return constraint(ctx, acts); }) |
                                ranges::view::filter([](const auto& result) { return result.has_value(); }) |
                                ranges::view::take(1),
                              HardRouteResult{},
                              [](const auto& acc, const auto& v) { return std::make_optional(v.value()); });
  }

  /// Checks soft route constraints and aggregates associated penalties.
  Cost soft(const InsertionRouteContext& ctx, const Activities& acts) const {
    return ranges::accumulate(ranges::view::all(softRouteConstraints_) |
                                ranges::view::transform([&](const auto& constraint) { return constraint(ctx, acts); }),
                              0.0);
  }

  // endregion

private:
  std::vector<HardRoute> hardRouteConstraints_;
  std::vector<SoftRoute> softRouteConstraints_;
  std::vector<HardActivity> hardActivityConstraints_;
  std::vector<SoftActivity> softActivityConstraints_;
};

}  // namespace vrp::algorithms::construction
