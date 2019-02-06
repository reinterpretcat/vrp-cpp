#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Constraints.hpp"
#include "models/extensions/problem/Helpers.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace vrp::algorithms::construction {

/// Checks whether vehicle can handle activity's demand.
/// Size can be interpreted as vehicle capacity change after visiting specific activity.
template<typename Size>
struct VehicleActivitySize final
  : public HardRouteConstraint
  , public HardActivityConstraint {
  /// Represents capacity type.
  using Capacity = Size;

  /// Represents job demand, both static and dynamic.
  struct Demand final {
    /// Keeps static and dynamic pickup amount.
    std::pair<Size, Size> pickup;
    /// Keeps static and dynamic delivery amount.
    std::pair<Size, Size> delivery;

    /// Returns size change as difference between pickup and delivery.
    Size change() const { return pickup.first + pickup.second - delivery.first - delivery.second; }
  };

  inline static const std::string StateKeyDemand = "demand";
  inline static const std::string StateKeyCapacity = "capacity";
  inline static const std::string StateKeyCurrent = "size_current";
  inline static const std::string StateKeyMaxFuture = "size_max_future";
  inline static const std::string StateKeyMaxPast = "size_max_past";

  explicit VehicleActivitySize(int code = 2) : code_(code) {}

  /// Accept route and updates its insertion state.
  void accept(InsertionRouteContext& context) const override {
    using namespace ranges;

    const auto& route = *context.route;
    auto tour = view::concat(view::single(route.start), route.tour.activities(), view::single(route.end));

    // calculate what must to be loaded at start
    auto start = ranges::accumulate(tour, Size{}, [&](const auto& acc, const auto& a) {
      return a->service.has_value() ? acc + getDemand(a).delivery.first : acc;
    });

    // determine actual load at each activity and max load in past
    auto end = ranges::accumulate(tour, std::pair<Size, Size>{start, start}, [&](const auto& acc, const auto& a) {
      auto current = acc.first + getDemand(a).change();
      auto max = std::max(acc.second, current);

      context.state->put<Size>(StateKeyCurrent, a, current);
      context.state->put<Size>(StateKeyMaxPast, a, max);

      return std::pair<Size, Size>{current, max};
    });

    // determine max load in future
    ranges::accumulate(tour | view::reverse, end.first, [&](const auto& acc, const auto& a) {
      auto max = std::max(acc, context.state->get<Size>(StateKeyCurrent, a).value());
      context.state->put<Size>(StateKeyMaxFuture, a, max);
      return max;
    });
  }

  /// Checks whether proposed vehicle and job can be used within route without violating size constraints.
  HardRouteConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                   const HardRouteConstraint::Job& job) const override {
    return models::problem::analyze_job<bool>(  //
             job,
             [&](const std::shared_ptr<const models::problem::Service>& service) {
               return canHandleSize(routeCtx, routeCtx.route->start, getDemand(service));
             },
             [&](const std::shared_ptr<const models::problem::Sequence>& sequence) {
               // TODO we can check at least static pickups/deliveries
               return true;
             })
      ? HardRouteConstraint::Result{}
      : HardRouteConstraint::Result{code_};
  }

  /// Checks whether proposed activity insertion doesn't violate size constraints.
  HardActivityConstraint::Result hard(const InsertionRouteContext& rCtx,
                                      const InsertionActivityContext& aCtx) const override {
    return canHandleSize(rCtx, aCtx.prev, getDemand(aCtx.target)) ? success() : stop(code_);
  }

  /// Returns a capacity size associated within vehicle.
  static Capacity getCapacity(const std::shared_ptr<const models::problem::Vehicle>& vehicle) {
    auto capacity = vehicle->dimens.find(StateKeyCapacity);
    return capacity != vehicle->dimens.end() ? std::any_cast<Size>(capacity->second) : Size{};
  }

  /// Returns demand associated service or empty if it does not exist.
  static Demand getDemand(const std::shared_ptr<const models::problem::Service>& service) {
    auto demand = service->dimens.find(StateKeyDemand);
    return demand != service->dimens.end() ? std::any_cast<Demand>(demand->second) : Demand{};
  }

  /// Returns demand associated within activity or empty if it does not exist.
  static Demand getDemand(const models::solution::Tour::Activity& activity) {
    return activity->service.has_value() ? getDemand(activity->service.value()) : Demand{};
  }

private:
  int code_;

  /// Estimates whether given size can be loaded into vehicle after pivot activity in tour.
  bool canHandleSize(const InsertionRouteContext& routeCtx,
                     const models::solution::Tour::Activity& pivot,
                     const Demand& demand) const {
    auto capacity = getCapacity(routeCtx.route->actor->vehicle);

    // cannot handle more static deliveries
    auto past = routeCtx.state->get<Size>(StateKeyMaxPast, pivot).value_or(Size{});
    if (past + demand.delivery.first > capacity) return false;

    // cannot handle more static pickups
    auto future = routeCtx.state->get<Size>(StateKeyMaxFuture, pivot).value_or(Size{});
    if (future + demand.pickup.first > capacity) return false;

    // can load more at current
    auto current = routeCtx.state->get<Size>(StateKeyCurrent, pivot).value_or(Size{});
    return current + demand.change() < capacity;
  }
};
}
