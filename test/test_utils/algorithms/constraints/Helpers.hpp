#pragma once

#include "models/problem/Fleet.hpp"
#include "test_utils/models/Factories.hpp"

namespace vrp::test {

inline algorithms::construction::HardActivityConstraint::Result
success() {
  return {};
}
inline algorithms::construction::HardActivityConstraint::Result
fail() {
  return {{true, 1}};
}
inline algorithms::construction::HardActivityConstraint::Result
stop() {
  return {{false, 1}};
}

inline std::shared_ptr<models::solution::Actor>
getActor(const std::string& id, const models::problem::Fleet& fleet) {
  auto vehicle = fleet.vehicle(id);
  auto detail = vehicle->details.front();
  return std::make_shared<models::solution::Actor>(
    models::solution::Actor{vehicle, DefaultDriver, detail.start, detail.end, detail.time});
}

inline models::problem::VehicleDetail
asDetail(const models::common::Location start,
         const std::optional<models::common::Location>& end,
         const models::common::TimeWindow time) {
  return models::problem::VehicleDetail{start, end, time};
}

inline std::vector<models::problem::VehicleDetail>
asDetails(const models::common::Location start,
          const std::optional<models::common::Location>& end,
          const models::common::TimeWindow time) {
  return {models::problem::VehicleDetail{start, end, time}};
}
}