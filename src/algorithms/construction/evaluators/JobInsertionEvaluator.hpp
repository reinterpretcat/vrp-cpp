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

    /// Activity departure time.
    models::common::Timestamp departure = 0;

    /// Activity detail.
    ActivityDetail detail;

    /// Creates a new context.
    static EvaluationContext empty(const models::common::Cost& cost, const models::common::Timestamp& departure) {
      return {false, 0, 0, cost, departure, {}};
    }

    /// Creates a new context from old one when insertion failed.
    static EvaluationContext fail(std::tuple<bool, int> error,
                                  const models::common::Timestamp& departure,
                                  const EvaluationContext& other) {
      return {std::get<0>(error), std::get<1>(error), other.index, other.cost, departure, other.detail};
    }

    /// Creates a new context from old one when insertion worse.
    static EvaluationContext skip(const models::common::Timestamp& departure, const EvaluationContext& other) {
      return {other.isStopped, other.code, other.index, other.cost, departure, other.detail};
    }

    /// Creates a new context.
    static EvaluationContext success(size_t index,
                                     const models::common::Cost& cost,
                                     const models::common::Timestamp& departure,
                                     const ActivityDetail& detail) {
      return {false, 0, index, cost, departure, detail};
    }

    /// Checks whether insertion is found.
    bool isSuccess() const { return cost < std::numeric_limits<models::common::Cost>::max(); }
  };

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

  /// Calculates activity insertion costs locally, i.e. by comparing extra costs of
  /// insertion the new activity k between activity i and j.
  /// Additional costs are then basically calculated as delta c = c_ik + c_kj - c_ij.
  models::common::Cost activityCosts(const InsertionRouteContext& routeCtx,
                                     const InsertionActivityContext& actCtx,
                                     const InsertionProgress& progress) const {
    using namespace vrp::models::common;
    using namespace vrp::models::solution;

    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = *actCtx.next;
    const auto& route = routeCtx.route.first;

    auto [tpCostLeft, actCostLeft, depTimeLeft] = analyze(*routeCtx.actor, prev, target, actCtx.departure);

    auto [tpCostRight, actCostRight, depTimeRight] = analyze(*routeCtx.actor, target, next, depTimeLeft);

    auto newCosts = tpCostLeft + tpCostRight + progress.completeness * (actCostLeft + actCostRight);

    if (routeCtx.route.first->tour.empty()) return newCosts;

    auto [tpCostOld, actCostOld, depTimeOld] =
      analyze(*route->actor,
              prev.type == Activity::Type::Start ? *route->start : prev,
              next.type == Activity::Type::End ? *route->end : next,
              prev.type == Activity::Type::Start ? route->start->schedule.departure : prev.schedule.departure);

    auto oldCosts = tpCostOld + progress.completeness * actCostOld;

    return newCosts - oldCosts;
  }

  /// Returns departure time from end activity taking into account time: departure time from start activity.
  models::common::Duration departure(const models::solution::Actor& actor,
                                     const models::solution::Activity& start,
                                     const models::solution::Activity& end,
                                     const models::common::Timestamp& depTime) const {
    auto arrival =
      depTime + transportCosts_->duration(actor.vehicle->profile, start.detail.location, end.detail.location, depTime);
    return std::max(arrival, end.detail.time.start) + activityCosts_->duration(actor, end, arrival);
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

  /// Analyzes route leg.
  std::tuple<Cost, Cost, Timestamp> analyze(const models::solution::Actor& actor,
                                            const models::solution::Activity& start,
                                            const models::solution::Activity& end,
                                            models::common::Timestamp time) const {
    auto arrival =
      time + transportCosts_->duration(actor.vehicle->profile, start.detail.location, end.detail.location, time);
    auto departure = std::max(arrival, end.detail.time.start) + activityCosts_->duration(actor, end, arrival);

    auto transportCost = transportCosts_->cost(actor, start.detail.location, end.detail.location, time);
    auto activityCost = activityCosts_->cost(actor, end, arrival);

    return std::make_tuple(transportCost, activityCost, departure);
  }

  std::shared_ptr<const models::costs::TransportCosts> transportCosts_;
  std::shared_ptr<const models::costs::ActivityCosts> activityCosts_;
};
}
