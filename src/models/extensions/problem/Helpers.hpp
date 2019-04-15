#pragma once

#include "models/problem/Driver.hpp"
#include "models/problem/Job.hpp"
#include "models/problem/Vehicle.hpp"

#include <range/v3/utility/variant.hpp>

namespace vrp::models::problem {

/// Analyzes job.
template<typename Result, typename ServiceFunc, typename SequenceFunc>
Result
analyze_job(const models::problem::Job& job, ServiceFunc&& serviceFun, SequenceFunc&& sequenceFunc) {
  if (job.index() == 0) return serviceFun(ranges::get<0>(job));

  return sequenceFunc(ranges::get<1>(job));
}

/// Creates job from service.
inline Job
as_job(const std::shared_ptr<const Service>& service) {
  return {ranges::emplaced_index<0>, service};
}

/// Creates job from sequence.
inline Job
as_job(const std::shared_ptr<const Sequence>& sequence) {
  return {ranges::emplaced_index<1>, sequence};
}

/// Returns id as string from dimens.
inline std::string
getId(const common::Dimensions& dimens) {
  return std::any_cast<std::string>(dimens.find("id")->second);
}

/// Returns id of the job.
struct get_job_id final {
  std::string operator()(const Job& job) const {
    return analyze_job<std::string>(
      job,
      [](const std::shared_ptr<const Service>& service) { return getId(service->dimens); },
      [](const std::shared_ptr<const Sequence>& sequence) { return getId(sequence->dimens); });
  }
};

/// Returns id of a vehicle.
struct get_vehicle_id final {
  std::string operator()(const Vehicle& vehicle) const { return getId(vehicle.dimens); }
};

/// Returns id of a driver.
struct get_driver_id final {
  std::string operator()(const Driver& driver) const { return getId(driver.dimens); }
};
}
