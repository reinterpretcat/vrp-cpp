#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "models/extensions/problem/Helpers.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::construction {

/// Sorts jobs in random order.
struct random_jobs_sorter final {
  void operator()(InsertionContext& ctx) const { ctx.random->shuffle(ctx.jobs.begin(), ctx.jobs.end()); }
};

/// Sorts jobs based on their size.
template<typename Size>
struct sized_jobs_sorter final {
  void operator()(InsertionContext& ctx) const {
    ranges::action::sort(
      ctx.jobs, [](const auto& lhs, const auto& rhs) { return cumulativeDemand(lhs) > cumulativeDemand(rhs); });
  }

private:
  static Size cumulativeDemand(const models::problem::Job& job) {
    auto demand = models::problem::analyze_job<typename VehicleActivitySize<Size>::Demand>(
      job,
      [](const std::shared_ptr<const models::problem::Service>& service) {
        return VehicleActivitySize<Size>::getDemand(service);
      },
      [](const std::shared_ptr<const models::problem::Sequence>& sequence) {
        // NOTE we use demand from the first service, this works best
        // only for pickup and delivery cases with two services in total.
        return VehicleActivitySize<Size>::getDemand(sequence->services.front());
      });

    return demand.pickup.first + demand.pickup.second + demand.delivery.first + demand.delivery.second;
  }
};
}
