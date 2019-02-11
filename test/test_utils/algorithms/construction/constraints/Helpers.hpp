#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionRouteContext.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/problem/Fleet.hpp"
#include "test_utils/models/Factories.hpp"

#include <test_utils/models/Helpers.hpp>

namespace vrp::test {

constexpr int StartActivityIndex = -1;
constexpr int EndActivityIndex = -2;

inline models::solution::Tour::Activity
getActivity(const algorithms::construction::InsertionRouteContext& ctx, int index) {
  if (index == StartActivityIndex) return ctx.route->start;
  if (index == EndActivityIndex) return ctx.route->end;

  return ctx.route->tour.get(static_cast<size_t>(index));
}

inline std::shared_ptr<models::solution::Actor>
getActor(const std::string& id, const models::problem::Fleet& fleet) {
  auto vehicle = find_vehicle_by_id{}(fleet, id);
  auto detail = vehicle->details.front();
  return std::make_shared<models::solution::Actor>(
    models::solution::Actor{vehicle, DefaultDriver, detail.start, detail.end, detail.time});
}

inline models::problem::Vehicle::Detail
asDetail(const models::common::Location start,
         const std::optional<models::common::Location>& end,
         const models::common::TimeWindow time) {
  return models::problem::Vehicle::Detail{start, end, time};
}

inline std::vector<models::problem::Vehicle::Detail>
asDetails(const models::common::Location start,
          const std::optional<models::common::Location>& end,
          const models::common::TimeWindow time) {
  return {models::problem::Vehicle::Detail{start, end, time}};
}


template<typename Base, typename Return, typename Arg1, typename Arg2>
struct SoftFunctionWrapper : Base {
  explicit SoftFunctionWrapper(typename Base::CheckFunc func) : func_(std::move(func)) {}

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(vrp::algorithms::construction::InsertionRouteContext& context) const override {}

  Return soft(const Arg1& arg1, const Arg2& arg2) const override { return func_(arg1, arg2); }

  typename Base::CheckFunc func_;
};

template<typename Base, typename Return, typename Arg1, typename Arg2>
struct HardFunctionWrapper : Base {
  explicit HardFunctionWrapper(typename Base::CheckFunc func) : func_(std::move(func)) {}

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(vrp::algorithms::construction::InsertionRouteContext& context) const override {}

  Return hard(const Arg1& arg1, const Arg2& arg2) const override { return func_(arg1, arg2); }

  typename Base::CheckFunc func_;
};

using HardRouteWrapper = HardFunctionWrapper<vrp::algorithms::construction::HardRouteConstraint,
                                             vrp::algorithms::construction::HardRouteConstraint::Result,
                                             vrp::algorithms::construction::InsertionRouteContext,
                                             vrp::algorithms::construction::HardRouteConstraint::Job>;

using SoftRouteWrapper = SoftFunctionWrapper<vrp::algorithms::construction::SoftRouteConstraint,
                                             vrp::models::common::Cost,
                                             vrp::algorithms::construction::InsertionRouteContext,
                                             vrp::algorithms::construction::HardRouteConstraint::Job>;

using HardActivityWrapper = HardFunctionWrapper<vrp::algorithms::construction::HardActivityConstraint,
                                                vrp::algorithms::construction::HardActivityConstraint::Result,
                                                vrp::algorithms::construction::InsertionRouteContext,
                                                vrp::algorithms::construction::InsertionActivityContext>;

using SoftActivityWrapper = SoftFunctionWrapper<vrp::algorithms::construction::SoftActivityConstraint,
                                                vrp::models::common::Cost,
                                                vrp::algorithms::construction::InsertionRouteContext,
                                                vrp::algorithms::construction::InsertionActivityContext>;
}
