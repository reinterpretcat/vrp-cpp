#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Fleets.hpp"
#include "algorithms/construction/extensions/States.hpp"
#include "utils/extensions/Variant.hpp"

namespace vrp::algorithms::construction {

/// Checks whether vehicle can handle activity of given size.
template<typename Size>
struct VehicleActivitySize final
  : public HardRouteConstraint
  , public HardActivityConstraint {
  inline static const std::string StateKey = "size";
  inline static const std::string StateKeyStart = StateKey + "_start";
  inline static const std::string StateKeyEnd = StateKey + "_end";

  VehicleActivitySize(const std::shared_ptr<const models::problem::Fleet>& fleet, int code = 2) : code_(code) {
    ranges::for_each(empty_actors(*fleet), [&](const auto& d) {
      auto key = actorSharedKey(StateKey, {{}, {}, d.start, d.end, d.time});
    });
  }

  /// Accept route and updates its insertion state.
  void accept(const models::solution::Route& route, InsertionRouteState& state) const override {
    // TODO
    // Save parameters for all possible actors
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

  int code_;
};
}
