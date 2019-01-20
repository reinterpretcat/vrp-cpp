#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Constraints.hpp"
#include "utils/extensions/Variant.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>

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
  void accept(InsertionRouteContext& context) const override {
    using namespace ranges;

    const auto& route = *context.route;

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

      context.state->put<Size>(StateKeyCurrent, *a, current);
      context.state->put<Size>(StateKeyMaxPast, *a, max);

      return std::pair<Size, Size>{current, max};
    });

    // determine max load in future
    ranges::accumulate(tour | view::reverse, Size{}, [&](const auto& acc, const auto& a) {
      auto max = std::max(acc, context.state->get<Size>(StateKeyCurrent, *a).value());
      context.state->put<Size>(StateKeyMaxFuture, *a, max);
      return max;
    });
  }

  /// Checks whether proposed vehicle and job can be used within route without violating size constraints.
  HardRouteConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                   const HardRouteConstraint::Job& job) const override {
    return utils::mono_result<bool>(job.visit(ranges::overload(
             [&](const std::shared_ptr<const models::problem::Service>& service) {
               return getSize(service) <= getSize(routeCtx.route->actor->vehicle);
             },
             [&](const std::shared_ptr<const models::problem::Shipment>& shipment) {
               return getSize(routeCtx.route->actor->vehicle) <= getSize(shipment);
             })))
      ? HardRouteConstraint::Result{}
      : HardRouteConstraint::Result{code_};
  }

  /// Checks whether proposed activity insertion doesn't violate size constraints.
  HardActivityConstraint::Result hard(const InsertionRouteContext& rCtx,
                                      const InsertionActivityContext& aCtx) const override {
    auto size = getSize(aCtx.target);
    auto base = rCtx.state->get<Size>(size < 0 ? StateKeyMaxPast : StateKeyMaxFuture, *aCtx.prev).value_or(Size{});
    auto value = size < 0 ? base - size : base + size;

    return value <= getSize(rCtx.route->actor->vehicle) ? success() : stop(code_);
  }

  /// Returns size of an entity (vehicle or job).
  template<typename T>
  static Size getSize(const std::shared_ptr<const T>& holder) {
    return std::any_cast<Size>(holder->dimens.find(StateKey)->second);
  }

  /// Returns size of activity.
  static Size getSize(const models::solution::Tour::Activity& activity) {
    return activity->type == models::solution::Activity::Type::Job && activity->job.has_value()
      ? utils::mono_result<Size>(activity->job.value().visit(ranges::overload(
          [&](const std::shared_ptr<const models::problem::Service>& service) { return getSize(service); },
          [&](const std::shared_ptr<const models::problem::Shipment>& shipment) { return getSize(shipment); })))
      : Size{};
  }

private:
  int code_;
};
}
