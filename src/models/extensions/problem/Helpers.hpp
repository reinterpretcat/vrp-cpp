#pragma once

#include "models/problem/Job.hpp"

#include <range/v3/utility/variant.hpp>

namespace vrp::models::problem {

/// Analyzes job.
template<typename Result, typename ServiceFunc, typename ShipmentFunc>
Result
analyze_job(const models::problem::Job& job, ServiceFunc&& serviceFun, ShipmentFunc&& shipmentFunc) {
  return job.index() == 0 ? serviceFun(ranges::get<0>(job)) : shipmentFunc(ranges::get<1>(job));
}
}
