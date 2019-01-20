#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/solution/Activity.hpp"
#include "utils/extensions/Ranges.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <range/v3/all.hpp>
#include <set>
#include <vector>

namespace vrp::algorithms::construction {

/// Specifies a base constraint behavior.
struct Constraint {
  /// Accept route and updates its state to allow more efficient constraint checks.
  /// Called in thread-safe context, so it is a chance to apply some changes.
  virtual void accept(InsertionRouteContext& context) const = 0;

  virtual ~Constraint() = default;
};

/// Specifies hard constraint which operates on route level.
struct HardRouteConstraint : virtual public Constraint {
  /// Specifies single hard constraint result type.
  using Result = std::optional<int>;
  /// Job alias.
  using Job = models::problem::Job;
  /// Specifies check function type.
  using CheckFunc = std::function<Result(const InsertionRouteContext&, const Job&)>;

  virtual Result hard(const InsertionRouteContext&, const Job&) const = 0;
};

/// Specifies soft constraint which operates on route level.
struct SoftRouteConstraint : virtual public Constraint {
  /// Job alias.
  using Job = models::problem::Job;

  /// Specifies check function type.
  using CheckFunc = std::function<models::common::Cost(const InsertionRouteContext&, const Job&)>;

  virtual models::common::Cost soft(const InsertionRouteContext&, const Job&) const = 0;
};

/// Specifies hard constraint which operation on activity level.
struct HardActivityConstraint : virtual public Constraint {
  /// Specifies single activity constraint result: int for code, bool to continue
  using Result = std::optional<std::tuple<bool, int>>;
  /// Specifies check function type.
  using CheckFunc = std::function<Result(const InsertionRouteContext&, const InsertionActivityContext&)>;

  virtual Result hard(const InsertionRouteContext&, const InsertionActivityContext&) const = 0;
};

/// Specifies soft constraint which operation on activity level.
struct SoftActivityConstraint : virtual public Constraint {
  /// Specifies check function type.
  using CheckFunc = std::function<models::common::Cost(const InsertionRouteContext&, const InsertionActivityContext&)>;

  virtual models::common::Cost soft(const InsertionRouteContext&, const InsertionActivityContext&) const = 0;
};

/// An insertion constraint which encapsulates behaviour of all possible constraint types.
class InsertionConstraint final {
  template<typename Base, typename Return, typename Arg1, typename Arg2>
  struct SoftFunctionWrapper : Base {
    explicit SoftFunctionWrapper(typename Base::CheckFunc func) : func_(std::move(func)) {}
    void accept(InsertionRouteContext& context) const override {}
    Return soft(const Arg1& arg1, const Arg2& arg2) const override { return func_(arg1, arg2); }
    typename Base::CheckFunc func_;
  };

  template<typename Base, typename Return, typename Arg1, typename Arg2>
  struct HardFunctionWrapper : Base {
    explicit HardFunctionWrapper(typename Base::CheckFunc func) : func_(std::move(func)) {}
    void accept(InsertionRouteContext& context) const override {}
    Return hard(const Arg1& arg1, const Arg2& arg2) const override { return func_(arg1, arg2); }
    typename Base::CheckFunc func_;
  };

public:
  // region Acceptance

  /// Accepts route and recalculates its states.
  void accept(InsertionRouteContext& context) const {
    ranges::for_each(constraints_, [&](const auto& c) { c->accept(context); });
  }

  // endregion

  // region Add route

  /// Adds hard route constraints
  InsertionConstraint& addHardRoute(std::shared_ptr<HardRouteConstraint> constraint) {
    constraints_.insert(constraint);
    hardRouteConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds hard route constraints
  InsertionConstraint& addHardRoute(HardRouteConstraint::CheckFunc constraint) {
    using Wrapper = HardFunctionWrapper<HardRouteConstraint,
                                        HardRouteConstraint::Result,
                                        InsertionRouteContext,
                                        HardRouteConstraint::Job>;

    auto c = std::make_shared<Wrapper>(std::move(constraint));
    constraints_.insert(c);
    hardRouteConstraints_.push_back(c);
    return *this;
  }

  /// Adds soft route constraint.
  InsertionConstraint& addSoftRoute(std::shared_ptr<SoftRouteConstraint> constraint) {
    constraints_.insert(constraint);
    softRouteConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds soft route constraint.
  InsertionConstraint& addSoftRoute(SoftRouteConstraint::CheckFunc constraint) {
    using Wrapper =
      SoftFunctionWrapper<SoftRouteConstraint, models::common::Cost, InsertionRouteContext, HardRouteConstraint::Job>;

    auto c = std::make_shared<Wrapper>(std::move(constraint));
    constraints_.insert(c);
    softRouteConstraints_.push_back(c);
    return *this;
  }

  // endregion

  // region Add activity

  /// Adds hard activity constraint.
  InsertionConstraint& addHardActivity(std::shared_ptr<HardActivityConstraint> constraint) {
    constraints_.insert(constraint);
    hardActivityConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds hard activity constraint.
  InsertionConstraint& addHardActivity(HardActivityConstraint::CheckFunc constraint) {
    using Wrapper = HardFunctionWrapper<HardActivityConstraint,
                                        HardActivityConstraint::Result,
                                        InsertionRouteContext,
                                        InsertionActivityContext>;

    auto c = std::make_shared<Wrapper>(std::move(constraint));
    constraints_.insert(c);
    hardActivityConstraints_.push_back(c);
    return *this;
  }

  /// Adds soft activity constraint.
  InsertionConstraint& addSoftActivity(std::shared_ptr<SoftActivityConstraint> constraint) {
    constraints_.insert(constraint);
    softActivityConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds soft activity constraint.
  InsertionConstraint& addSoftActivity(SoftActivityConstraint::CheckFunc constraint) {
    using Wrapper = SoftFunctionWrapper<SoftActivityConstraint,
                                        models::common::Cost,
                                        InsertionRouteContext,
                                        InsertionActivityContext>;

    auto c = std::make_shared<Wrapper>(std::move(constraint));
    constraints_.insert(c);
    softActivityConstraints_.push_back(c);
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
    constraints_.insert(v);
    hardRouteConstraints_.push_back(v);
    hardActivityConstraints_.push_back(v);
    return *this;
  }

  template<typename T>
  InsertionConstraint& addSoft(
    std::shared_ptr<typename std::enable_if<std::is_base_of<SoftRouteConstraint, T>::value &&
                                              std::is_base_of<SoftActivityConstraint, T>::value,
                                            T>::type> v) {
    constraints_.insert(v);
    softRouteConstraints_.push_back(v);
    softActivityConstraints_.push_back(v);
    return *this;
  }

  // endregion

  // region Add all

  template<typename T>
  InsertionConstraint& add(
    std::shared_ptr<typename std::enable_if<
      std::is_base_of<HardRouteConstraint, T>::value && std::is_base_of<HardActivityConstraint, T>::value &&
        std::is_base_of<SoftRouteConstraint, T>::value && std::is_base_of<SoftActivityConstraint, T>::value,
      T>::type> v) {
    constraints_.insert(v);
    hardRouteConstraints_.push_back(v);
    hardActivityConstraints_.push_back(v);
    softRouteConstraints_.push_back(v);
    softActivityConstraints_.push_back(v);
    return *this;
  }

  // endregion

  // region Route level evaluations

  /// Checks whether all hard route constraints are fulfilled.
  /// Returns the code of first failed constraint or empty value.
  HardRouteConstraint::Result hard(const InsertionRouteContext& ctx, const HardRouteConstraint::Job& job) const {
    return utils::accumulate_while(ranges::view::all(hardRouteConstraints_),
                                   HardRouteConstraint::Result{},
                                   [](const auto& r) { return !r.has_value(); },
                                   [&](const auto&, const auto& constraint) { return constraint->hard(ctx, job); });
  }

  /// Checks soft route constraints and aggregates associated penalties.
  models::common::Cost soft(const InsertionRouteContext& ctx, const HardRouteConstraint::Job& job) const {
    return ranges::accumulate(
      ranges::view::all(softRouteConstraints_), models::common::Cost{}, [&](const auto& acc, const auto& constraint) {
        return acc + constraint->soft(ctx, job);
      });
  }

  // endregion

  // region Activity level evaluations

  HardActivityConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                      const InsertionActivityContext& actCtx) const {
    return utils::accumulate_while(
      ranges::view::all(hardActivityConstraints_),
      HardActivityConstraint::Result{},
      [](const auto& r) { return !r.has_value(); },
      [&](const auto&, const auto& constraint) { return constraint->hard(routeCtx, actCtx); });
  }

  models::common::Cost soft(const InsertionRouteContext& routeCtx, const InsertionActivityContext& actCtx) const {
    return ranges::accumulate(
      ranges::view::all(softActivityConstraints_),
      models::common::Cost{},
      [&](const auto& acc, const auto& constraint) { return acc + constraint->soft(routeCtx, actCtx); });
  }

  // endregion

private:
  std::vector<std::shared_ptr<HardRouteConstraint>> hardRouteConstraints_;
  std::vector<std::shared_ptr<SoftRouteConstraint>> softRouteConstraints_;
  std::vector<std::shared_ptr<HardActivityConstraint>> hardActivityConstraints_;
  std::vector<std::shared_ptr<SoftActivityConstraint>> softActivityConstraints_;
  std::set<std::shared_ptr<Constraint>> constraints_;
};

}  // namespace vrp::algorithms::construction
