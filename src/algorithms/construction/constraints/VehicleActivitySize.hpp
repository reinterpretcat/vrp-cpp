#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/States.hpp"
#include "utils/extensions/Variant.hpp"

#include <array>

namespace vrp::algorithms::construction {

/// Checks whether vehicle can handle activity of given size.
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

    // set loads for each activity and start/end
    auto ends = ranges::accumulate(tour, std::array<Size, 4>{}, [&](const auto& acc, const auto& a) {
      auto size = getSize(a);

      auto current = acc[0] + size;
      auto max = std::max(acc[1], current);

      auto [start, end] = a->job.has_value() && a->job.value().index() == 0
        ? std::pair<Size, Size>{acc[2] + (size > 0 ? size : Size{}),  //
                                acc[3] - (size > 0 ? size : Size{})}
        : std::pair<Size, Size>{acc[2], acc[3]};

      state.put<Size>(StateKeyCurrent, *a, current);
      state.put<Size>(StateKeyMaxPast, *a, max);

      return std::array<Size, 4>{current, max, start, end};
    });

    state.put<Size>(StateKeyStart, ends[2]);
    state.put<Size>(StateKeyEnd, ends[3]);

    // set max future load on activity to prevent overload
    ranges::accumulate(tour | view::reverse, Size{}, [&](const auto& acc, const auto& a) {
      auto result = std::max(acc, state.get<Size>(StateKeyCurrent, *a).value());
      state.put<Size>(StateKeyMaxFuture, *a, result);
      return result;
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

  inline Size getSize(const models::solution::Tour::Activity& activity) const {
    return activity->job.has_value() ? getSize(activity->job.value()) : Size{};
  }

  template<typename T>
  inline Size getSize(const std::shared_ptr<const T>& holder) const {
    return std::any_cast<Size>(holder->dimens.find(StateKey)->second);
  }

  int code_;
};
}
