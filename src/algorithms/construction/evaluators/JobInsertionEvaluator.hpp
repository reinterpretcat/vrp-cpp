#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

namespace vrp::algorithms::construction {

struct JobInsertionEvaluator {
  JobInsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts,
                        std::shared_ptr<const models::costs::ActivityCosts> activityCosts) :
    transportCosts_(std::move(transportCosts)),
    activityCosts_(std::move(activityCosts)) {}

  virtual ~JobInsertionEvaluator() = default;

protected:
  using ActivityDetail = models::solution::Activity::Detail;

  /// Specifies evaluation context.
  struct EvaluationContext final {
    /// True, if processing has to be stopped.
    bool isStopped;
    /// Violation code.
    int code = 0;
    /// Insertion index.
    size_t index = 0;
    /// Best cost.
    models::common::Cost cost = std::numeric_limits<models::common::Cost>::max();

    /// Activity detail.
    ActivityDetail detail;

    /// Creates a new context.
    static EvaluationContext empty(const models::common::Cost& cost) { return {false, 0, 0, cost, {}}; }

    /// Creates a new context from old one when insertion failed.
    static EvaluationContext fail(std::tuple<bool, int> error, const EvaluationContext& other) {
      return {std::get<0>(error), std::get<1>(error), other.index, other.cost /*, departure*/, other.detail};
    }

    /// Creates a new context from old one when insertion worse.
    static EvaluationContext skip(const EvaluationContext& other) {
      return {other.isStopped, other.code, other.index, other.cost, other.detail};
    }

    /// Creates a new context.
    static EvaluationContext success(size_t index, const models::common::Cost& cost, const ActivityDetail& detail) {
      return {false, 0, index, cost, detail};
    }

    /// Checks whether insertion is found.
    bool isSuccess() const { return cost < std::numeric_limits<models::common::Cost>::max(); }
  };

  /// TODO move to timing constraint
  /// Calculates vehicle specific costs.
  models::common::Cost vehicleCosts(const InsertionRouteContext& ctx) const {
    models::common::Cost deltaFirst = 0.0;
    models::common::Cost deltaLast = 0.0;

    const auto& route = ctx.route.first;

    if (!route->tour.empty()) {
      double firstCostNew = transportCosts_->cost(*ctx.actor,
                                                  ctx.actor->detail.start,               //
                                                  route->tour.first()->detail.location,  //
                                                  ctx.departure);
      double firstCostOld = transportCosts_->cost(*route->actor,
                                                  route->start->detail.location,         //
                                                  route->tour.first()->detail.location,  //
                                                  route->start->schedule.departure);

      deltaFirst = firstCostNew - firstCostOld;

      if (ctx.actor->detail.end.has_value()) {
        auto last = route->tour.last();
        auto lastDepartureOld = last->schedule.departure;
        auto lastDepartureNew = std::max(models::common::Timestamp{0},  //
                                         lastDepartureOld + (ctx.departure - route->start->schedule.departure));

        auto lastCostNew = transportCosts_->cost(*ctx.actor,
                                                 last->detail.location,          //
                                                 ctx.actor->detail.end.value(),  //
                                                 lastDepartureNew);
        auto lastCostOld = transportCosts_->cost(*route->actor,
                                                 last->detail.location,        //
                                                 route->end->detail.location,  //
                                                 lastDepartureNew);

        deltaLast = lastCostNew - lastCostOld;
      }
    }

    return deltaFirst + deltaLast;
  }

  /// Returns departure time from end activity taking into account time: departure time from start activity.
  models::common::Duration departure(const models::solution::Actor& actor,
                                     const models::solution::Activity& start,
                                     const models::solution::Activity& end,
                                     const models::common::Timestamp& depTime) const {
    auto arrival =
      depTime + transportCosts_->duration(actor.vehicle->profile, start.detail.location, end.detail.location, depTime);
    auto workStart = std::max(arrival, end.detail.time.start);
    return workStart + activityCosts_->duration(actor, end, workStart);
  }

  InsertionResult failure(const EvaluationContext& eCtx) const { return make_result_failure(eCtx.code); }

  InsertionResult success(const EvaluationContext& e,
                          const InsertionRouteContext& i,
                          const models::solution::Tour::Activity& a) const {
    return make_result_success({e.cost, a->job.value(), {{a, e.index}}, i.actor, i.route, i.departure});
  }

private:
  using Cost = models::common::Cost;
  using Timestamp = models::common::Timestamp;

  std::shared_ptr<const models::costs::TransportCosts> transportCosts_;
  std::shared_ptr<const models::costs::ActivityCosts> activityCosts_;
};
}
