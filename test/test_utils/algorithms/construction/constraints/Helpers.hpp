#pragma once

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
}