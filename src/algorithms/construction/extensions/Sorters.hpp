#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "models/extensions/problem/Helpers.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::construction {

/// Sorts jobs in random order.
struct random_jobs_sorter final {
  void operator()(InsertionContext& ctx) const {
    ctx.random->shuffle(ctx.solution->required.begin(), ctx.solution->required.end());
  }
};

/// Sorts jobs based on their size.
template<typename Size>
struct sized_jobs_sorter final {
  bool isDesc = true;
  void operator()(InsertionContext& ctx) const {
    ranges::action::sort(ctx.solution->required, [&](const auto& lhs, const auto& rhs) {
      return isDesc ? cumulativeDemand(lhs) > cumulativeDemand(rhs) : cumulativeDemand(lhs) < cumulativeDemand(rhs);
    });
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

/// Sorts jobs by their distance to vehicle start locations.
struct ranked_jobs_sorter final {
  bool isDesc = true;

  void operator()(InsertionContext& ctx) const {
    ranges::action::sort(ctx.solution->required, [&](const auto& lhs, const auto& rhs) {
      auto left = distanceToStart(ctx, lhs);
      auto right = distanceToStart(ctx, rhs);

      return isDesc ? left > right : left < right;
    });
  }

private:
  static models::common::Distance distanceToStart(const InsertionContext& ctx, const models::problem::Job& job) {
    return ranges::min(ranges::view::for_each(ctx.problem->fleet->profiles(), [&](const auto& profile) {
      return ranges::yield(ctx.problem->jobs->rank(profile, job));
    }));
  }
};
}
