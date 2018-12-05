#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/States.hpp"
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
  inline static const std::string StateKeyStart = StateKey + "_start";
  inline static const std::string StateKeyEnd = StateKey + "_end";

  explicit VehicleActivitySize(int code = 2) : code_(code) {}

  /// Accept route and updates its insertion state.
  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {
    using namespace ranges;

    auto tour = view::concat(view::single(route.start), route.tour.activities(), view::single(route.end));

    // calculate what must to be loaded at start and what has to be brought to the end
    auto ends = ranges::accumulate(tour, std::pair<Size, Size>{}, [&](const auto& acc, const auto& a) {
      auto size = getSize(a);
      return a->job.has_value() && a->job.value().index() == 0
        ? std::pair<Size, Size>{acc.first - (size < 0 ? size : Size{}),  //
                                acc.second + (size > 0 ? size : Size{})}
        : acc;
    });

    state.put<Size>(StateKeyStart, ends.first);
    state.put<Size>(StateKeyEnd, ends.second);

    // determine actual load at each activity and max load in past
    ranges::accumulate(tour, std::pair<Size, Size>{ends.first, ends.first}, [&](const auto& acc, const auto& a) {
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
    auto max = getSize(routeCtx.actor->vehicle);
    auto min = Size{};

    auto start = getState(StateKeyStart, routeCtx).value_or(max);
    auto end = getState(StateKeyEnd, routeCtx).value_or(min);

    return utils::mono_result<bool>(job.visit(ranges::overload(
             [&](const std::shared_ptr<const models::problem::Service>& service) {
               auto size = getSize(service);
               return size > 0 ? start - size >= min : end - size <= max;
             },
             [&](const std::shared_ptr<const models::problem::Shipment>& shipment) {
               // TODO check that total sum is good and each does not violate
               return true;
             })))
      ? HardRouteConstraint::Result{}
      : HardRouteConstraint::Result{code_};
  }

  /// Checks whether proposed activity insertion doesn't violate size constraints.
  HardActivityConstraint::Result check(const InsertionRouteContext& routeCtx,
                                       const InsertionActivityContext& actCtx) const override {
    // TODO
    return {};
  }

private:
  inline std::optional<Size> getState(const std::string& key, const InsertionRouteContext& routeCtx) const {
    return routeCtx.route.second->get<Size>(actorSharedKey(key, *routeCtx.actor));
  }

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
