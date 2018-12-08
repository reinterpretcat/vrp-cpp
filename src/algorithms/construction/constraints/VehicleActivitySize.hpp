#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Constraints.hpp"
#include "utils/extensions/Variant.hpp"

namespace vrp::algorithms::construction {

/// Checks whether vehicle can handle activity of given size.
/// Assume that "negative" size is delivery (unloaded from vehicle),
/// positive is pickup (loaded to vehicle).
template<typename Size>
struct VehicleActivitySize final
  : public HardRouteConstraint
  , public HardActivityConstraint {
  inline static const std::string StateKey = "size";
  inline static const std::string StateKeyCurrent = StateKey + "_current";
  inline static const std::string StateKeyMaxFuture = StateKey + "_max_future";
  inline static const std::string StateKeyMaxPast = StateKey + "_max_past";

  explicit VehicleActivitySize(int code = 2) : code_(code) {}

  /// Accept route and updates its insertion state.
  void accept(models::solution::Route& route, InsertionRouteState& state) const override {
    using namespace ranges;

    auto tour = view::concat(view::single(route.start), route.tour.activities(), view::single(route.end));

    // calculate what must to be loaded at start
    auto start = ranges::accumulate(tour, Size{}, [&](const auto& acc, const auto& a) {
      auto size = getSize(a);
      return a->job.has_value() && a->job.value().index() == 0 ? acc - (size < 0 ? size : Size{}) : acc;
    });

    // determine actual load at each activity and max load in past
    ranges::accumulate(tour, std::pair<Size, Size>{start, start}, [&](const auto& acc, const auto& a) {
      auto size = getSize(a);
      auto current = acc.first + size;
      auto max = std::max(acc.second, current);

      state.put<Size>(StateKeyCurrent, *a, current);
      state.put<Size>(StateKeyMaxPast, *a, max);

      return std::pair<Size, Size>{current, max};
    });

    // determine max load in future
    ranges::accumulate(tour | view::reverse, Size{}, [&](const auto& acc, const auto& a) {
      auto max = std::max(acc, state.get<Size>(StateKeyCurrent, *a).value());
      state.put<Size>(StateKeyMaxFuture, *a, max);
      return max;
    });
  }

  /// Checks whether proposed vehicle and job can be used within route without violating size constraints.
  HardRouteConstraint::Result check(const InsertionRouteContext& routeCtx,
                                    const HardRouteConstraint::Job& job) const override {
    return utils::mono_result<bool>(job.visit(ranges::overload(
             [&](const std::shared_ptr<const models::problem::Service>& service) {
               return getSize(service) <= getSize(routeCtx.actor->vehicle);
             },
             [&](const std::shared_ptr<const models::problem::Shipment>& shipment) {
               return getSize(routeCtx.actor->vehicle) <= getSize(shipment);
             })))
      ? HardRouteConstraint::Result{}
      : HardRouteConstraint::Result{code_};
  }

  /// Checks whether proposed activity insertion doesn't violate size constraints.
  HardActivityConstraint::Result check(const InsertionRouteContext& routeCtx,
                                       const InsertionActivityContext& actCtx) const override {
    const auto& state = routeCtx.route.second;

    auto size = getSize(actCtx.target);
    auto base = state->get<Size>(size < 0 ? StateKeyMaxPast : StateKeyMaxFuture, *actCtx.prev).value_or(Size{});
    auto value = size < 0 ? base - size : base + size;

    return value <= getSize(routeCtx.actor->vehicle) ? success() : stop(code_);
  }

private:
  template<typename T>
  inline Size getSize(const std::shared_ptr<const T>& holder) const {
    return std::any_cast<Size>(holder->dimens.find(StateKey)->second);
  }

  inline Size getSize(const models::solution::Tour::Activity& activity) const {
    return activity->type == models::solution::Activity::Type::Job && activity->job.has_value()
      ? utils::mono_result<Size>(activity->job.value().visit(ranges::overload(
          [&](const std::shared_ptr<const models::problem::Service>& service) { return getSize(service); },
          [&](const std::shared_ptr<const models::problem::Shipment>& shipment) { return getSize(shipment); })))
      : Size{};
  }

  int code_;
};
}