#pragma once

#include "algorithms/construction/insertion/InsertionRouteContext.hpp"
#include "models/common/Cost.hpp"
#include "models/costs/TransportCosts.hpp"

#include <algorithm>

namespace vrp::algorithms::construction {

struct JobInsertionEvaluator {
  explicit JobInsertionEvaluator(std::shared_ptr<const models::costs::TransportCosts> transportCosts) :
    transportCosts_(std::move(transportCosts)) {}

  virtual ~JobInsertionEvaluator() = default;

protected:
  /// Estimates vehicle switch cost if when new vehicle is used.
  models::common::Cost vehicleSwitchCost(const InsertionRouteContext& ctx) const {
    models::common::Cost deltaFirst = 0.0;
    models::common::Cost deltaLast = 0.0;

    if (!ctx.route->tour.empty()) {
      double firstCostNew = transportCosts_->cost(*ctx.actor,
                                                  ctx.actor->vehicle->start,               //
                                                  ctx.route->tour.first()->stop.location,  //
                                                  ctx.time);
      double firstCostOld = transportCosts_->cost(ctx.route->actor,
                                                  ctx.route->start.location,               //
                                                  ctx.route->tour.first()->stop.location,  //
                                                  ctx.route->start.schedule.departure);

      deltaFirst = firstCostNew - firstCostOld;

      if (ctx.actor->vehicle->end.has_value()) {
        auto last = ctx.route->tour.last();
        auto lastDepartureOld = last->stop.schedule.departure;
        auto lastDepartureNew = std::max(models::common::Timestamp{0},  //
                                         lastDepartureOld + (ctx.time - ctx.route->start.schedule.departure));

        auto lastCostNew = transportCosts_->cost(*ctx.actor,
                                                 last->stop.location,              //
                                                 ctx.actor->vehicle->end.value(),  //
                                                 lastDepartureNew);
        auto lastCostOld = transportCosts_->cost(ctx.route->actor,
                                                 last->stop.location,      //
                                                 ctx.route->end.location,  //
                                                 lastDepartureNew);

        deltaLast = lastCostNew - lastCostOld;
      }
    }

    return deltaFirst + deltaLast;
  }

private:
  std::shared_ptr<const models::costs::TransportCosts> transportCosts_;
};
}
