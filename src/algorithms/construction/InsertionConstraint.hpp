#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/InsertionRouteContext.hpp"
#include "algorithms/construction/InsertionSolutionContext.hpp"
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
  /// Accepts insertion solution context allowing to update job insertion data.
  /// Called in thread-safe context.
  virtual void accept(InsertionSolutionContext&) const = 0;

  /// Accept route and updates its state to allow more efficient constraint checks.
  /// Called in thread-safe context, so it is a chance to apply some changes.
  virtual void accept(InsertionRouteContext&) const = 0;

  /// Returns unique constraint state keys.
  virtual ranges::any_view<int> stateKeys() const = 0;

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

///// An insertion constraint which encapsulates behaviour of all possible constraint types.
class InsertionConstraint final {
public:
  // region Acceptance

  /// Accepts context.
  void accept(InsertionSolutionContext& ctx) const {
    ranges::for_each(constraints_, [&](const auto& c) { c->accept(ctx); });
  }

  /// Accepts route and recalculates its states.
  void accept(InsertionRouteContext& ctx) const {
    ranges::for_each(constraints_, [&](const auto& c) { c->accept(ctx); });
  }

  // endregion

  // region Add generic

  /// Adds generic constraint.
  InsertionConstraint& add(std::shared_ptr<Constraint> constraint) {
    addStateKeys(constraint->stateKeys());
    constraints_.push_back(constraint);
    return *this;
  }

  // endregion

  // region Add route

  /// Adds hard route constraints
  InsertionConstraint& addHardRoute(std::shared_ptr<HardRouteConstraint> constraint) {
    addStateKeys(constraint->stateKeys());
    constraints_.push_back(constraint);
    hardRouteConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds soft route constraint.
  InsertionConstraint& addSoftRoute(std::shared_ptr<SoftRouteConstraint> constraint) {
    addStateKeys(constraint->stateKeys());
    constraints_.push_back(constraint);
    softRouteConstraints_.push_back(std::move(constraint));
    return *this;
  }

  // endregion

  // region Add activity

  /// Adds hard activity constraint.
  InsertionConstraint& addHardActivity(std::shared_ptr<HardActivityConstraint> constraint) {
    addStateKeys(constraint->stateKeys());
    constraints_.push_back(constraint);
    hardActivityConstraints_.push_back(std::move(constraint));
    return *this;
  }

  /// Adds soft activity constraint.
  InsertionConstraint& addSoftActivity(std::shared_ptr<SoftActivityConstraint> constraint) {
    addStateKeys(constraint->stateKeys());
    constraints_.push_back(constraint);
    softActivityConstraints_.push_back(std::move(constraint));
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
    addStateKeys(v->stateKeys());
    constraints_.push_back(v);
    hardRouteConstraints_.push_back(v);
    hardActivityConstraints_.push_back(v);
    return *this;
  }

  template<typename T>
  InsertionConstraint& addSoft(
    std::shared_ptr<typename std::enable_if<std::is_base_of<SoftRouteConstraint, T>::value &&
                                              std::is_base_of<SoftActivityConstraint, T>::value,
                                            T>::type> v) {
    addStateKeys(v->stateKeys());
    constraints_.push_back(v);
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
    addStateKeys(v->stateKeys());
    constraints_.push_back(v);
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
  void addStateKeys(ranges::any_view<int> keys) {
    ranges::for_each(keys, [&](int key) {
      if (stateKeys_.find(key) != stateKeys_.end()) {
        throw std::invalid_argument("Constraint state key clash: " + std::to_string(key));
      }
      stateKeys_.insert(key);
    });
  }

  std::vector<std::shared_ptr<HardRouteConstraint>> hardRouteConstraints_;
  std::vector<std::shared_ptr<SoftRouteConstraint>> softRouteConstraints_;
  std::vector<std::shared_ptr<HardActivityConstraint>> hardActivityConstraints_;
  std::vector<std::shared_ptr<SoftActivityConstraint>> softActivityConstraints_;
  std::vector<std::shared_ptr<Constraint>> constraints_;
  std::set<int> stateKeys_;
};

}  // namespace vrp::algorithms::construction
