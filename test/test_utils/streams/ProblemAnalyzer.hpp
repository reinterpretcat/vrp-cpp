#pragma once

#include "models/problem/Job.hpp"
#include "models/problem/Jobs.hpp"
#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"

#include <range/v3/all.hpp>

namespace vrp::test {

models::problem::Job
getJobAt(size_t index, const models::problem::Jobs& jobs) {
  auto v = jobs.all() | ranges::to_vector;
  return v.at(index);
}

std::shared_ptr<const models::problem::Vehicle>
getVehicleAt(size_t index, const models::problem::Fleet& fleet) {
  auto v = fleet.vehicles() | ranges::to_vector;
  return v.at(index);
}

std::shared_ptr<const models::problem::Driver>
getDriverAt(size_t index, const models::problem::Fleet& fleet) {
  auto v = fleet.drivers() | ranges::to_vector;
  return v.at(index);
}

}