#pragma once

#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/refinement/logging/LogToExtras.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/extensions/solution/Helpers.hpp"
#include "streams/in/json/HereProblemJson.hpp"
#include "utils/Date.hpp"

#include <any>
#include <nlohmann/json.hpp>
#include <optional>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::streams::out {

namespace detail::here {

// region Statistic

struct Timing final {
  int driving;
  int serving;
  int waiting;
  int breakTime;
};

struct Statistic final {
  double cost;
  int distance;
  int duration;
  Timing times;
};

inline void
to_json(nlohmann::json& j, const Timing& timing) {
  j["driving"] = timing.driving;
  j["serving"] = timing.serving;
  j["waiting"] = timing.waiting;
  j["break"] = timing.breakTime;
}

inline void
to_json(nlohmann::json& j, const Statistic& statistic) {
  j = nlohmann::json{{"cost", statistic.cost},
                     {"distance", statistic.distance},
                     {"duration", statistic.duration},
                     {"times", statistic.times}};
}

// endregion

// region Tour

struct Schedule final {
  std::string arrival;
  std::string departure;
};

struct Interval final {
  std::string start;
  std::string end;
};

struct Activity final {
  std::string jobId;
  std::string type;
  std::optional<std::vector<double>> location;
  std::optional<Interval> time;
  std::optional<std::string> jobTag;
};

struct Stop final {
  std::vector<double> location;
  Schedule time;
  std::vector<int> load;
  std::vector<Activity> activities;
};

struct Tour final {
  std::string vehicleId;
  std::string typeId;
  std::vector<Stop> stops;
  Statistic statistic;
};

inline void
to_json(nlohmann::json& j, const Schedule& schedule) {
  j["arrival"] = schedule.arrival;
  j["departure"] = schedule.departure;
}

inline void
to_json(nlohmann::json& j, const Interval& interval) {
  j["start"] = interval.start;
  j["end"] = interval.end;
}

inline void
to_json(nlohmann::json& j, const Activity& activity) {
  j["jobId"] = activity.jobId;
  j["type"] = activity.type;
  if (activity.location) j["location"] = activity.location.value();
  if (activity.time) j["time"] = activity.time.value();
  if (activity.jobTag) j["jobTag"] = activity.jobTag.value();
}

inline void
to_json(nlohmann::json& j, const Stop& stop) {
  j = nlohmann::json{
    {"location", stop.location}, {"time", stop.time}, {"load", stop.load}, {"activities", stop.activities}};
}

inline void
to_json(nlohmann::json& j, const Tour& tour) {
  j = nlohmann::json{
    {"vehicleId", tour.vehicleId}, {"typeId", tour.typeId}, {"stops", tour.stops}, {"statistic", tour.statistic}};
}

// endregion

// region Unassigned

struct UnassignedJobReason final {
  int code;
  std::string description;
};

struct UnassignedJob final {
  std::string jobId;
  std::vector<UnassignedJobReason> reasons;
};

inline void
to_json(nlohmann::json& j, const UnassignedJobReason& reason) {
  j["code"] = reason.code;
  j["description"] = reason.description;
}

inline void
to_json(nlohmann::json& j, const UnassignedJob& job) {
  j = nlohmann::json{{"jobId", job.jobId}, {"reasons", job.reasons}};
}

// endregion

// region Extras

/// Defines iteration model.
struct Iteration final {
  /// Iteration number.
  int number;
  /// Best known cost
  double cost;
  /// Elapsed time in seconds.
  double timestamp;
  /// Amount of tours
  int tours;
  /// Amount of unassigned jobs.
  int unassinged;
};

/// Contains extra information.
struct Extras final {
  /// Stores information about iteration performance.
  std::vector<Iteration> performance;
};

inline void
to_json(nlohmann::json& j, const Iteration& iteration) {
  j = nlohmann::json{{"number", iteration.number},
                     {"cost", iteration.cost},
                     {"timestamp", iteration.timestamp},
                     {"tours", iteration.tours},
                     {"unassinged", iteration.unassinged}};
}

inline void
to_json(nlohmann::json& j, const Extras& extras) {
  j = nlohmann::json{{"performance", extras.performance}};
}

// endregion

// region Common

struct Solution final {
  std::string problemId;
  Statistic statistic;
  std::vector<Tour> tours;
  std::vector<UnassignedJob> unassigned;
  Extras extras;
};

inline void
to_json(nlohmann::json& j, const Solution& solution) {
  j = nlohmann::json{{"problemId", solution.problemId}, {"statistic", solution.statistic}, {"tours", solution.tours}};

  if (!solution.extras.performance.empty()) j["extras"] = solution.extras;
  if (!solution.unassigned.empty()) j["unassigned"] = solution.unassigned;
}

// endregion

// region Calculations

struct Leg final {
  models::common::Location location;
  models::common::Timestamp departure;
  models::common::Distance distance;
  models::common::Duration duration;
  Timing timing;
  models::common::Cost cost;
  int load;
};

struct create_solution final {
  Solution operator()(const models::Problem& problem, const models::EstimatedSolution& es) const {
    using namespace ranges;
    using namespace algorithms::construction;
    using namespace models::common;

    // TODO split method

    const auto& coordIndex = std::any_cast<streams::in::CoordIndex>(problem.extras->at("coordIndex"));
    auto date = utils::timestamp_to_rc3339_string{};
    auto solution = Solution{};

    solution.problemId = std::any_cast<std::string>(problem.extras->at("problemId"));

    solution.statistic = ranges::accumulate(
      view::zip(es.first->routes, view::iota(0)),
      Statistic{0, 0, 0, Timing{0, 0, 0, 0}},
      [&](const auto& acc, const auto& indexedRoute) {
        auto [route, routeIndex] = indexedRoute;
        const auto& actor = *route->actor;

        auto load = ranges::accumulate(route->tour.activities(), 0, [&](const auto& acc, const auto& a) {
          return a->service.has_value() ? acc + VehicleActivitySize<int>::getDemand(a).delivery.first : acc;
        });

        auto tour = Tour{
          {},
          {},
          {Stop{coordIndex.find(route->tour.start()->detail.location),
                Schedule{date(route->tour.start()->schedule.arrival), date(route->tour.start()->schedule.departure)},
                {load},
                {Activity{"departure", "departure", {}, {}}}}},
          {}};

        auto leg = ranges::accumulate(
          view::zip(route->tour.activities() | view::drop(1), view::iota(1)),
          Leg{route->tour.start()->detail.location,
              route->tour.start()->schedule.departure,
              0,
              0,
              Timing{0, 0, 0, 0},
              0,
              load},
          [&](const auto& acc, const auto& indexedActivity) {
            auto [act, activityIndex] = indexedActivity;

            // get activity type
            auto type = getActivityType(*act, activityIndex);
            bool isBreak = type == "break";
            auto jobId = type == "pickup" || type == "delivery"
              ? models::problem::get_job_id{}(models::problem::as_job(act->service.value()))
              : type;

            // timings
            auto driving =
              problem.transport->duration(actor.vehicle->profile, acc.location, act->detail.location, acc.departure);
            auto arrival = acc.departure + driving;
            auto start = std::max(act->schedule.arrival, act->detail.time.start);
            auto waiting = start - act->schedule.arrival;
            auto serving = problem.activity->duration(actor, *act, act->schedule.arrival);
            auto departure = start + serving;

            bool isSameLocation = acc.location == act->detail.location;

            if (!isSameLocation)
              tour.stops.push_back(
                Stop{coordIndex.find(act->detail.location), Schedule{date(arrival), date(departure)}, {acc.load}, {}});

            auto load = changeLoad(acc.load, VehicleActivitySize<int>::getDemand(act));
            auto addOptionalFields = tour.stops.back().activities.size() > 1;

            tour.stops.back().time.departure = date(departure);
            tour.stops.back().load[0] = load;
            tour.stops.back().activities.push_back(
              Activity{jobId,
                       type,
                       addOptionalFields ? std::make_optional(coordIndex.find(act->detail.location))
                                         : std::optional<std::vector<double>>{},
                       addOptionalFields ? std::make_optional(Interval{date(arrival), date(departure)})
                                         : std::optional<Interval>{},
                       getActivityTag(*act)});

            auto cost = problem.activity->cost(actor, *act, act->schedule.arrival) +
              problem.transport->cost(actor, acc.location, act->detail.location, acc.departure);

            return Leg{
              act->detail.location,
              act->schedule.departure,
              acc.distance +
                problem.transport->distance(actor.vehicle->profile, acc.location, act->detail.location, acc.departure),
              acc.duration + departure - acc.departure,
              Timing{static_cast<int>(acc.timing.driving + driving),
                     static_cast<int>(acc.timing.serving + (isBreak ? 0 : serving)),
                     static_cast<int>(acc.timing.waiting + waiting),
                     static_cast<int>(acc.timing.breakTime + (isBreak ? serving : 0))},
              acc.cost + cost,
              load};
          });

        tour.vehicleId = std::any_cast<std::string>(actor.vehicle->dimens.at("id"));
        tour.typeId = std::any_cast<std::string>(actor.vehicle->dimens.at("typeId"));
        tour.statistic = Statistic{leg.cost + actor.vehicle->costs.fixed,
                                   static_cast<int>(leg.distance),
                                   static_cast<int>(leg.duration),
                                   leg.timing};

        solution.tours.push_back(tour);

        return Statistic{acc.cost + leg.cost + actor.vehicle->costs.fixed,
                         static_cast<int>(acc.distance + leg.distance),
                         static_cast<int>(acc.duration + leg.duration),
                         Timing{acc.times.driving + leg.timing.driving,
                                acc.times.serving + leg.timing.serving,
                                acc.times.waiting + leg.timing.waiting,
                                acc.times.breakTime + leg.timing.breakTime}};
      });

    if (!es.first->unassigned.empty()) solution.unassigned = getUnassigned(es);

    solution.extras = getExtras(es);

    return std::move(solution);
  }

private:
  static std::vector<UnassignedJob> getUnassigned(const models::EstimatedSolution& es) {
    using namespace ranges;
    using namespace nlohmann;

    return view::for_each(es.first->unassigned,
                          [](const auto& un) {
                            return yield(UnassignedJob{models::problem::get_job_id{}(un.first),
                                                       {UnassignedJobReason{mapUnassignedCode(un.second),
                                                                            mapUnassignedDescription(un.second)}}});
                          }) |
      to_vector;
  }

  static std::string getActivityType(const models::solution::Activity& activity, int index) {
    if (activity.service.has_value()) {
      auto dim = activity.service.value()->dimens.find("type");
      if (dim != activity.service.value()->dimens.end() && std::any_cast<std::string>(dim->second) == "break")
        return "break";

      auto demand = std::any_cast<algorithms::construction::VehicleActivitySize<int>::Demand>(
        activity.service.value()->dimens.find("demand")->second);

      return demand.pickup.first > 0 || demand.pickup.second > 0 ? "pickup" : "delivery";
    }

    return index == 0 ? "departure" : "arrival";
  }

  static std::optional<std::string> getActivityTag(const models::solution::Activity& activity) {
    if (activity.service.has_value()) {
      auto dim = activity.service.value()->dimens.find("tag");
      if (dim != activity.service.value()->dimens.end()) return {std::any_cast<std::string>(dim->second)};
    }
    return {};
  }

  static int mapUnassignedCode(int code) {
    using Codes = in::HereProblemUnassignedCodes;

    switch (code) {
      case Codes::Time:
        return 2;
      case Codes::Size:
        return 3;
      case Codes::DistanceLimit:
        return 101;
      case Codes::DurationLimit:
        return 102;
      case Codes::Skill:
        return 1;
      case Codes::Reachable:
        return 100;
      default:
        return 0;
    }
  }

  static std::string mapUnassignedDescription(int code) {
    using Codes = in::HereProblemUnassignedCodes;

    switch (code) {
      case Codes::Time:
        return "cannot be visited within time window";
      case Codes::Size:
        return "does not fit into any vehicle due to capacity";
      case Codes::DistanceLimit:
        return "cannot be assigned due to max distance constraint of vehicle";
      case Codes::DurationLimit:
        return "cannot be assigned due to shift time constraint of vehicle";
      case Codes::Skill:
        return "cannot serve required skill";
      case Codes::Reachable:
        return "location unreachable";
      default:
        return "unknown";
    }
  }

  static int changeLoad(int current, const algorithms::construction::VehicleActivitySize<int>::Demand& demand) {
    return current - demand.delivery.first - demand.delivery.second + demand.pickup.first + demand.pickup.second;
  }

  static Extras getExtras(const models::EstimatedSolution& es) {
    using namespace algorithms::refinement;

    if (es.first->extras) {
      auto iterPair = es.first->extras->find(log_to_extras::ExtrasKey);
      if (iterPair != es.first->extras->end()) {
        return Extras{ranges::accumulate(std::any_cast<const std::vector<log_to_extras::Iteration>&>(iterPair->second),
                                         std::vector<Iteration>{},
                                         [](auto& acc, const auto& iter) {
                                           acc.push_back(Iteration{
                                             static_cast<int>(iter.number),
                                             iter.cost.actual,
                                             static_cast<double>(iter.timestamp) / 1000,
                                             static_cast<int>(iter.routes),
                                             static_cast<int>(iter.unassigned),
                                           });
                                           return std::move(acc);
                                         })};
      }
    }

    return {};
  }
};

// endregion
}

struct dump_solution_as_here_json final {
  std::shared_ptr<const models::Problem> problem;
  void operator()(std::ostream& out, const models::EstimatedSolution& es) const {
    nlohmann::json json;
    detail::here::to_json(json, detail::here::create_solution{}(*problem, es));
    out << json.dump(4);
  }
};
}