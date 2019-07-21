#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/constraints/ConditionalJob.hpp"

#include <unordered_set>

namespace vrp::streams::in::detail::rich {

// TODO rename to RequirementsConstraint?
/// Represents capability constraint to much vehicle/driver to job by their tagged capabilities.
struct CapabilityConstraint final : public vrp::algorithms::construction::HardRouteConstraint {
  using RawType = std::unordered_set<std::string>;
  using WrappedType = std::shared_ptr<RawType>;
  using Result = vrp::algorithms::construction::HardRouteConstraint::Result;

  static inline const std::string Skills = "skills";
  static inline const std::string Facilities = "facilities";

  explicit CapabilityConstraint(int code) : code_(code) {}

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(vrp::algorithms::construction::InsertionSolutionContext&) const override {}

  void accept(vrp::algorithms::construction::InsertionRouteContext&) const override {}

  Result hard(const vrp::algorithms::construction::InsertionRouteContext& ctx,
              const vrp::models::problem::Job& job) const override {
    return models::problem::analyze_job<Result>(
      job,
      [&ctx, code = code_](const std::shared_ptr<const models::problem::Service>& service) {
        return check(*ctx.route->actor, service->dimens, code);
      },
      [&ctx, code = code_](const std::shared_ptr<const models::problem::Sequence>& sequence) {
        return check(*ctx.route->actor, sequence->dimens, code);
      });
  }

private:
  static Result check(const models::solution::Actor& actor, const models::common::Dimensions& required, int code) {
    return checkVehicle(*actor.vehicle, required) && checkDriver(*actor.driver, required) ? Result{} : Result{code};
  }

  static bool checkTags(const std::string& tag,
                        const vrp::models::common::Dimensions& target,
                        const vrp::models::common::Dimensions& required) {
    if (required.find(tag) == required.end()) return true;
    if (target.find(tag) == target.end()) return false;

    const auto& values = std::any_cast<const WrappedType&>(target.at(tag));

    return ranges::all_of(*std::any_cast<const WrappedType&>(required.at(tag)),
                          [&values](const auto& v) { return values->find(v) != values->end(); });
  }

  static bool checkVehicle(const models::problem::Vehicle& vehicle, const vrp::models::common::Dimensions& required) {
    return checkTags(Facilities, vehicle.dimens, required);
  }

  static bool checkDriver(const models::problem::Driver& driver, const vrp::models::common::Dimensions& required) {
    return checkTags(Skills, driver.dimens, required);
  }

  int code_;
};
}