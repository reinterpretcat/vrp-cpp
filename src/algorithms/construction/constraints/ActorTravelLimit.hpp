#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/extensions/Constraints.hpp"
#include "models/common/Distance.hpp"
#include "models/common/Duration.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Comparators.hpp"


namespace vrp::algorithms::construction {

/// Allows to assign jobs with some condition.
/// NOTE should be used after ActorActivityTiming.
struct ActorTravelLimit final : public HardActivityConstraint {
  constexpr static int BaseKey = 20;
  constexpr static int DistanceKey = BaseKey + 0;
  constexpr static int DurationKey = BaseKey + 1;

  constexpr static int DistanceCode = 5;
  constexpr static int DurationCode = 6;

  /// Specifies actor limit type.
  struct Limit final {
    using Distance = std::optional<models::common::Distance>;
    using Duration = std::optional<models::common::Duration>;

    /// Actor condition.
    std::function<bool(const models::solution::Actor&)> condition;

    /// Maximum distance driven by actor.
    Distance maxDistance;

    /// Maximum duration spent by actor.
    Duration maxDuration;
  };

  explicit ActorTravelLimit(const std::vector<Limit>& limits,
                            const std::shared_ptr<const models::costs::TransportCosts>& transport,
                            const std::shared_ptr<const models::costs::ActivityCosts>& activity) :
    actorLimits_(),
    initLimits_(limits),
    transport_(transport),
    activity_(activity) {}

  ranges::any_view<int> stateKeys() const override {
    using namespace ranges;
    return view::concat(view::single(DurationKey), view::single(DistanceKey));
  }

  void accept(InsertionSolutionContext& ctx) const override {
    if (!initLimits_.empty()) {
      ranges::for_each(initLimits_, [&](const auto& limit) {
        ranges::for_each(ctx.registry->all() | ranges::view::filter([&](const auto& a) { return limit.condition(*a); }),
                         [&](const auto& actor) {
                           assert(actorLimits_.find(actor) == actorLimits_.end());

                           actorLimits_[actor] = std::make_pair(limit.maxDistance, limit.maxDuration);
                         });
      });

      initLimits_.clear();
    }
  }

  void accept(InsertionRouteContext& ctx) const override {
    using namespace ranges;
    using namespace vrp::models::common;

    auto [loc, dep, distance, duration] = ranges::accumulate(
      ctx.route->tour.activities(),
      std::tuple{
        ctx.route->tour.start()->detail.location, ctx.route->tour.start()->schedule.departure, Distance{}, Duration{}},
      [&](const auto& acc, auto& a) {
        auto [loc, dep, totalDist, totalDur] = acc;

        totalDist += transport_->distance(ctx.route->actor->vehicle->profile, loc, a->detail.location, dep);
        totalDur += a->schedule.departure - dep;

        return std::tuple{a->detail.location, a->schedule.departure, totalDist, totalDur};
      });

    ctx.state->put<Distance>(DistanceKey, distance);
    ctx.state->put<Duration>(DurationKey, duration);
  }

  HardActivityConstraint::Result hard(const InsertionRouteContext& routeCtx,
                                      const InsertionActivityContext& actCtx) const override {
    auto limitPair = actorLimits_.find(routeCtx.route->actor);

    if (limitPair == actorLimits_.end()) return success();

    if (!checkDistance(routeCtx, actCtx, limitPair->second.first)) return stop(DistanceCode);

    return checkDuration(routeCtx, actCtx, limitPair->second.second) ? success() : stop(DurationCode);
  }

private:
  bool checkDistance(const InsertionRouteContext& routeCtx,
                     const InsertionActivityContext& actCtx,
                     Limit::Distance limit) const {
    if (!limit) return true;

    return checkTravel(
      routeCtx, actCtx, DistanceKey, limit.value(), [&](const auto& profile, auto from, auto to, auto dep) {
        return transport_->distance(profile, from, to, dep);
      });
  }

  bool checkDuration(const InsertionRouteContext& routeCtx,
                     const InsertionActivityContext& actCtx,
                     Limit::Duration limit) const {
    if (!limit) return true;

    // NOTE consider extra operation time
    auto maxValue = limit.value() - actCtx.target->detail.duration;

    return checkTravel(routeCtx, actCtx, DurationKey, maxValue, [&](const auto& profile, auto from, auto to, auto dep) {
      return transport_->duration(profile, from, to, dep);
    });
  }

  template<typename T, typename TravelFunc>
  bool checkTravel(const InsertionRouteContext& routeCtx,
                   const InsertionActivityContext& actCtx,
                   int key,
                   T limit,
                   TravelFunc func) const {
    const auto& actor = *routeCtx.route->actor;
    const auto& prev = *actCtx.prev;
    const auto& target = *actCtx.target;
    const auto& next = actCtx.next;

    auto current = routeCtx.state->get<T>(key).value_or(0);

    auto prevToTarget =
      func(actor.vehicle->profile, prev.detail.location, target.detail.location, prev.schedule.departure);

    if (!next) return current + prevToTarget < limit;

    auto prevToNext =
      func(actor.vehicle->profile, prev.detail.location, next.value()->detail.location, prev.schedule.departure);

    auto targetToNext =
      func(actor.vehicle->profile, target.detail.location, next.value()->detail.location, target.schedule.departure);

    return current + prevToTarget + targetToNext - prevToNext <= limit;
  }

  mutable std::unordered_map<std::shared_ptr<const models::solution::Actor>,  //
                             std::pair<Limit::Distance, Limit::Duration>>
    actorLimits_;

  mutable std::vector<Limit> initLimits_;

  std::shared_ptr<const models::costs::TransportCosts> transport_;
  std::shared_ptr<const models::costs::ActivityCosts> activity_;
};
}