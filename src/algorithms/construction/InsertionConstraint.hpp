#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/solution/Activity.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::algorithms::construction {

/// Specifies a base constraint behavior.
struct Constraint {
  /// Accept route and updates its state to allow more efficient constraint checks.
  virtual void accept(const models::solution::Route& route, InsertionRouteState& state) const = 0;

  virtual ~Constraint() = default;
};

/// Specifies hard constraint which operates on route level.
struct HardRouteConstraint : virtual public Constraint {
  /// Specifies single hard constraint result type.
  using Result = std::optional<int>;
  /// Specifies activity collection type.
  using Activities = ranges::any_view<const models::solution::Activity>;
  /// Specifies check function type.
  using CheckFunc = std::function<Result(const InsertionRouteContext&, const Activities&)>;

  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {}

  virtual Result check(const InsertionRouteContext&, const Activities&) const = 0;
};

/// Specifies soft constraint which operates on route level.
struct SoftRouteConstraint : virtual public Constraint {
  /// Specifies activity collection type.
  using Activities = ranges::any_view<const models::solution::Activity>;

  /// Specifies check function type.
  using CheckFunc = std::function<models::common::Cost(const InsertionRouteContext&, const Activities&)>;

  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {}

  virtual models::common::Cost check(const InsertionRouteContext&, const Activities&) const = 0;
};

/// Specifies hard constraint which operation on activity level.
struct HardActivityConstraint : virtual public Constraint {
  /// Specifies single activity constraint result: int for code, bool to continue
  using Result = std::optional<std::tuple<bool, int>>;
  /// Specifies check function type.
  using CheckFunc = std::function<Result(const InsertionRouteContext&, const InsertionActivityContext&)>;

  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {}

  virtual Result check(const InsertionRouteContext&, const InsertionActivityContext&) const = 0;
};

/// Specifies soft constraint which operation on activity level.
struct SoftActivityConstraint : virtual public Constraint {
  /// Specifies check function type.
  using CheckFunc = std::function<models::common::Cost(const InsertionRouteContext&, const InsertionActivityContext&)>;

  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {}

  virtual models::common::Cost check(const InsertionRouteContext&, const InsertionActivityContext&) const = 0;
};

/// An insertion constraint which encapsulates behaviour of all possible constraint types.
class InsertionConstraint final {
  template<typename Base, typename Return, typename Arg1, typename Arg2>
  struct CheckFunctionWrapper : Base {
    explicit CheckFunctionWrapper(typename Base::CheckFunc func) : func_(std::move(func)) {}
    Return check(const Arg1& arg1, const Arg2& arg2) const override { return func_(arg1, arg2); }
    typename Base::CheckFunc func_;
  };

public:
  // region Add route

  /// Adds hard route constraints
  InsertionConstraint& addHardRoute(std::shared_ptr<HardRouteConstraint> constraint) {
    hardRouteConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds hard route constraints
  InsertionConstraint& addHardRoute(HardRouteConstraint::CheckFunc constraint) {
    using Wrapper = CheckFunctionWrapper<HardRouteConstraint,
                                         HardRouteConstraint::Result,
                                         InsertionRouteContext,
                                         HardRouteConstraint::Activities>;

    hardRouteConstraints_.push_back(std::make_shared<Wrapper>(std::move(constraint)));
    return *this;
  }

  /// Adds soft route constraint.
  InsertionConstraint& addSoftRoute(std::shared_ptr<SoftRouteConstraint> constraint) {
    softRouteConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds soft route constraint.
  InsertionConstraint& addSoftRoute(SoftRouteConstraint::CheckFunc constraint) {
    using Wrapper = CheckFunctionWrapper<SoftRouteConstraint,
                                         models::common::Cost,
                                         InsertionRouteContext,
                                         HardRouteConstraint::Activities>;

    softRouteConstraints_.push_back(std::make_shared<Wrapper>(std::move(constraint)));
    return *this;
  }

  // endregion

  // region Add activity

  /// Adds hard activity constraint.
  InsertionConstraint& addHardActivity(std::shared_ptr<HardActivityConstraint> constraint) {
    hardActivityConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds hard activity constraint.
  InsertionConstraint& addHardActivity(HardActivityConstraint::CheckFunc constraint) {
    using Wrapper = CheckFunctionWrapper<HardActivityConstraint,
                                         HardActivityConstraint::Result,
                                         InsertionRouteContext,
                                         InsertionActivityContext>;

    hardActivityConstraints_.push_back(std::make_shared<Wrapper>(std::move(constraint)));
    return *this;
  }

  /// Adds soft activity constraint.
  InsertionConstraint& addSoftActivity(std::shared_ptr<SoftActivityConstraint> constraint) {
    softActivityConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds soft activity constraint.
  InsertionConstraint& addSoftActivity(SoftActivityConstraint::CheckFunc constraint) {
    using Wrapper = CheckFunctionWrapper<SoftActivityConstraint,
                                         models::common::Cost,
                                         InsertionRouteContext,
                                         InsertionActivityContext>;

    softActivityConstraints_.push_back(std::make_shared<Wrapper>(std::move(constraint)));
    return *this;
  }

  // endregion

  // region Add mixed

  /// Adds constraint as hard route and hard activity.
  template<typename T>
  InsertionConstraint& addHard(
    std::shared_ptr<typename std::enable_if<std::is_base_of<HardRouteConstraint, T>::value &&
                                              std::is_base_of<HardActivityConstraint, T>::value,
                                            T>::type> v) {
    hardRouteConstraints_.push_back(v);
    hardActivityConstraints_.push_back(v);
    return *this;
  }

  // endregion

  // region Route level evaluations

  /// Checks whether all hard route constraints are fulfilled.
  /// Returns the code of first failed constraint or empty value.
  HardRouteConstraint::Result hard(const InsertionRouteContext& ctx,
                                   const HardRouteConstraint::Activities& acts) const {
    return ranges::accumulate(
      ranges::view::all(hardRouteConstraints_) |
        ranges::view::transform([&](const auto& constraint) { return constraint->check(ctx, acts); }) |
        ranges::view::filter([](const auto& result) { return result.has_value(); }) | ranges::view::take(1),
      HardRouteConstraint::Result{},
      [](const auto& acc, const auto& v) { return std::make_optional(v.value()); });
  }

  /// Checks soft route constraints and aggregates associated penalties.
  models::common::Cost soft(const InsertionRouteContext& ctx, const HardRouteConstraint::Activities& acts) const {
    return ranges::accumulate(
      ranges::view::all(softRouteConstraints_) |
        ranges::view::transform([&](const auto& constraint) { return constraint->check(ctx, acts); }),
      0.0);
  }

  // endregion

  // region Activity level evaluations

  HardActivityConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                      const InsertionActivityContext& actCtx) const {
    return ranges::accumulate(
      ranges::view::all(hardActivityConstraints_) |
        ranges::view::transform([&](const auto& constraint) { return constraint->check(routeCtx, actCtx); }) |
        ranges::view::filter([](const auto& result) { return result.has_value(); }) | ranges::view::take(1),
      HardActivityConstraint::Result{},
      [](const auto& acc, const auto& v) { return std::make_optional(v.value()); });
  }

  models::common::Cost soft(const InsertionRouteContext& routeCtx, const InsertionActivityContext& actCtx) const {
    return ranges::accumulate(
      ranges::view::all(softActivityConstraints_) |
        ranges::view::transform([&](const auto& constraint) { return constraint->check(routeCtx, actCtx); }),
      0.0);
  }

  // endregion

private:
  std::vector<std::shared_ptr<HardRouteConstraint>> hardRouteConstraints_;
  std::vector<std::shared_ptr<SoftRouteConstraint>> softRouteConstraints_;
  std::vector<std::shared_ptr<HardActivityConstraint>> hardActivityConstraints_;
  std::vector<std::shared_ptr<SoftActivityConstraint>> softActivityConstraints_;
};

}  // namespace vrp::algorithms::construction
