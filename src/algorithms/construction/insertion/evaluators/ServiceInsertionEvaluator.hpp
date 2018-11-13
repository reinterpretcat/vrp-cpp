#pragma once

#include "algorithms/construction/insertion/InsertionConstraint.hpp"
#include "algorithms/construction/insertion/InsertionResult.hpp"
#include "algorithms/construction/insertion/evaluators/JobInsertionEvaluator.hpp"
#include "models/common/Cost.hpp"
#include "models/common/TimeWindow.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Service.hpp"
#include "models/solution/Stop.hpp"

#include <numeric>
#include <utility>

namespace vrp::algorithms::construction {

struct ServiceInsertionEvaluator final : private JobInsertionEvaluator {
  /// Keeps insertion context data.
  struct Context final {
    /// Insertion index.
    int index;
    /// Violated constraint codes.
    std::vector<int> violations;
    /// Service time windows.
    models::common::TimeWindow time;
  };

public:
  ServiceInsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                            std::shared_ptr<const InsertionConstraint> constraint) :
    JobInsertionEvaluator(std::move(transportCosts)),
    constraint_(std::move(constraint)) {}

  InsertionResult evaluate(const std::shared_ptr<const models::problem::Service>& service,
                           const InsertionRouteContext& ctx,
                           models::common::Cost bestKnown) const {
    auto activity = models::solution::build_activity{}            //
                      .withJob(models::problem::as_job(service))  //
                      .owned();

    // check hard constraints on route level.
    auto error = constraint_->hard(ctx, ranges::view::single(activity));
    if (error.has_value()) { return {ranges::emplaced_index<1>, InsertionFailure{error.value()}}; }

    // calculate additional costs on route level.
    auto additionalCosts = constraint_->soft(ctx, ranges::view::single(activity)) + vehicleSwitchCost(ctx);

    return {ranges::emplaced_index<1>, InsertionFailure{}};
  }

private:
  /// Analyzes tour trying to find best insertion index.
  Context analyze(models::solution::Activity& activity,
                  const InsertionRouteContext& ctx,
                  models::common::Cost bestKnown) {
    using namespace vrp::utils;
    using namespace vrp::models;

    auto [start, end] = waypoints(ctx);

    return {};
  };

  /// Creates start/end stops of vehicle.
  std::pair<models::solution::Stop, models::solution::Stop> waypoints(const InsertionRouteContext& ctx) {
    using namespace vrp::utils;
    using namespace vrp::models;

    // create start/end for new vehicle
    auto start = solution::build_stop{}
                   .withLocation(ctx.actor->vehicle->start)                                                        //
                   .withSchedule({ctx.actor->vehicle->time.start, std::numeric_limits<common::Timestamp>::max()})  //
                   .owned();
    auto end = solution::build_stop{}
                 .withLocation(ctx.actor->vehicle->end.value_or(ctx.actor->vehicle->start))  //
                 .withSchedule({0, ctx.actor->vehicle->time.end})                            //
                 .owned();

    return {start, end};
  }

  std::shared_ptr<const InsertionConstraint> constraint_;
};

}  // namespace vrp::algorithms::construction
