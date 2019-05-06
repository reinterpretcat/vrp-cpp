#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/constraints/ConditionalJob.hpp"

#include <unordered_set>

namespace vrp::streams::in::detail::here {

// region Constraints

/// Represents a break constraint to simulate vehicle breaks.
struct BreakConstraint final : public vrp::algorithms::construction::HardActivityConstraint {
  static constexpr int Code = 4;

  BreakConstraint() :
    conditionalJob_([](const vrp::algorithms::construction::InsertionSolutionContext& ctx,
                       const models::problem::Job& job) {
      return models::problem::analyze_job<bool>(
        job,
        [&ctx](const std::shared_ptr<const models::problem::Service>& service) {
          // mark service as ignored only if it has break type and vehicle id is not present in routes
          if (isNotBreak(service)) return true;

          const auto& vehicleId = std::any_cast<std::string>(service->dimens.at("vehicleId"));
          return ranges::find_if(ctx.routes, [&vehicleId, &ctx](const auto& iCtx) {
                   // TODO check arrival time at last activity to avoid assigning break as last
                   auto isTarget = std::any_cast<std::string>(iCtx.route->actor->vehicle->dimens.at("id")) == vehicleId;
                   return isTarget && !ctx.required.empty();
                 }) != ctx.routes.end();
        },
        [](const std::shared_ptr<const models::problem::Sequence>& sequence) { return true; });
    }) {}

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(vrp::algorithms::construction::InsertionSolutionContext& ctx) const override {
    conditionalJob_.accept(ctx);
  }

  void accept(vrp::algorithms::construction::InsertionRouteContext&) const override {}

  vrp::algorithms::construction::HardActivityConstraint::Result hard(
    const vrp::algorithms::construction::InsertionRouteContext& routeCtx,
    const vrp::algorithms::construction::InsertionActivityContext& actCtx) const override {
    using namespace vrp::algorithms::construction;
    // TODO check that break is not assigned as last?
    return isNotBreak(actCtx.target->service.value()) || actCtx.prev->service.has_value() ? success() : stop(Code);
  }

private:
  static bool isNotBreak(const std::shared_ptr<const models::problem::Service>& service) {
    auto type = service->dimens.find("type");
    if (type == service->dimens.end() || std::any_cast<std::string>(type->second) != "break") return true;

    return false;
  }

  vrp::algorithms::construction::ConditionalJob conditionalJob_;
};

/// Represents skill constraint to much vehicle to job by skills
struct SkillConstraint final : public vrp::algorithms::construction::HardRouteConstraint {
  using RawType = std::unordered_set<std::string>;
  using WrappedType = std::shared_ptr<RawType>;
  using Result = vrp::algorithms::construction::HardRouteConstraint::Result;

  static constexpr int Code = 10;

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(vrp::algorithms::construction::InsertionSolutionContext&) const override {}

  void accept(vrp::algorithms::construction::InsertionRouteContext&) const override {}

  Result hard(const vrp::algorithms::construction::InsertionRouteContext& ctx,
              const vrp::models::problem::Job& job) const override {
    return models::problem::analyze_job<Result>(
      job,
      [&ctx](const std::shared_ptr<const models::problem::Service>& service) {
        return satisfy(ctx.route->actor->vehicle->dimens, service->dimens) ? Result{} : Result{Code};
      },
      [&ctx](const std::shared_ptr<const models::problem::Sequence>& sequence) {
        return satisfy(ctx.route->actor->vehicle->dimens, sequence->dimens) ? Result{} : Result{Code};
      });
  }

  static bool satisfy(const vrp::models::common::Dimensions& target, const vrp::models::common::Dimensions& required) {
    if (required.find("skills") == required.end()) return true;
    if (target.find("skills") == target.end()) return false;

    const auto& skills = std::any_cast<const WrappedType&>(target.at("skills"));

    return ranges::all_of(*std::any_cast<const WrappedType&>(required.at("skills")),
                          [&skills](const auto& skill) { return skills->find(skill) != skills->end(); });
  }
};

/// Provides the way to check whether location is reachable.
/// Non-reachable locations are defined by max int constant.
struct ReachableConstraint final : public vrp::algorithms::construction::HardActivityConstraint {
  static constexpr int Code = 11;
  static constexpr double UnreachableDistance = std::numeric_limits<int>::max();

  explicit ReachableConstraint(const std::shared_ptr<const models::costs::TransportCosts>& transport) :
    transport_(transport) {}

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(vrp::algorithms::construction::InsertionSolutionContext& ctx) const override {}

  void accept(vrp::algorithms::construction::InsertionRouteContext&) const override {}

  vrp::algorithms::construction::HardActivityConstraint::Result hard(
    const vrp::algorithms::construction::InsertionRouteContext& routeCtx,
    const vrp::algorithms::construction::InsertionActivityContext& actCtx) const override {
    using namespace vrp::algorithms::construction;

    const auto& actor = *routeCtx.route->actor;
    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = actCtx.next;

    auto prevToTarget = transport_->distance(
      actor.vehicle->profile, prev.detail.location, target.detail.location, prev.schedule.departure);

    if (prevToTarget >= UnreachableDistance) return stop(Code);

    if (!next) return success();

    auto targetToNext = transport_->distance(
      actor.vehicle->profile, target.detail.location, next.value()->detail.location, target.schedule.departure);

    return targetToNext < UnreachableDistance ? success() : stop(Code);
  }

private:
  std::shared_ptr<const models::costs::TransportCosts> transport_;
};

// endregion
}