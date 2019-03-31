#pragma once

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
  //  j["cost"] = statistic.cost;
  //  j["distance"] = statistic.distance;
  //  j["duration"] = statistic.duration;
  //  to_json(j["times"], statistic.times);
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
}

inline void
to_json(nlohmann::json& j, const Stop& stop) {
  //  j["location"] = tour.location;
  //  j["time"] = tour.time;
  //  j["load"] = tour.load;
  //  j["activities"] = tour.activities;

  j = nlohmann::json{
    {"location", stop.location}, {"time", stop.time}, {"load", stop.load}, {"activities", stop.activities}};
}

inline void
to_json(nlohmann::json& j, const Tour& tour) {
  //  j["vehicleId"] = tour.vehicleId;
  //  j["typeId"] = tour.typeId;
  //  j["stops"] = tour.stops;
  //  j["statistic"] = tour.statistic;
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
  //  j["jobId"] = job.jobId;
  //  j["reasons"] = job.reasons;
  j = nlohmann::json{{"jobId", job.jobId}, {"reasons", job.reasons}};
}

// endregion

// region Common

struct Solution final {
  std::string problemId;
  Statistic statistic;
  std::vector<Tour> tours;
  std::vector<UnassignedJob> unassigned;
};

inline void
to_json(nlohmann::json& j, const Solution& solution) {
  //  j["problemId"] = solution.problemId;
  //  j["statistic"] = solution.statistic;
  //  j["tours"] = solution.tours;
  //  j["unassigned"] = solution.unassigned;
  j = nlohmann::json{{"problemId", solution.problemId},
                     {"statistic", solution.statistic},
                     {"tours", solution.tours},
                     {"unassigned", solution.unassigned}};
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
};

inline std::string
getActivityType(const models::solution::Activity& activity, int index) {
  if (activity.service.has_value()) {
    auto dim = activity.service.value()->dimens.find("type");
    if (dim != activity.service.value()->dimens.end() && std::any_cast<std::string>(dim) == "break") return "break";

    // TODO get demand and detect pickup/delivery

    return "";
  }

  return index == 0 ? "departure" : "arrival";
}

// inline bool
// isSameLocations(const std::vector<double>& lhs, const std::vector<double>& rhs) {
//  return lhs.at(0) != rhs.at(0) || lhs.at(1) != rhs.at(1)
//}

inline Solution
createSolution(const models::Problem& problem, const models::EstimatedSolution& es) {
  using namespace ranges;

  const auto& coordIndex = std::any_cast<streams::in::CoordIndex>(problem.extras->at("coordIndex"));
  auto dateConverter = utils::timestamp_to_rc3339_string{};
  auto solution = Solution{};

  solution.statistic = ranges::accumulate(
    view::zip(es.first->routes, view::iota(0)),
    Statistic{es.second.total(), 0, 0, Timing{0, 0, 0, 0}},
    [&](const auto& acc, const auto& indexedRoute) {
      auto [route, routeIndex] = indexedRoute;
      const auto& actor = *route->actor;

      auto tour = Tour{};

      auto leg = ranges::accumulate(  //
        view::zip(route->tour.activities(), view::iota(0)),
        Leg{route->tour.start()->detail.location, route->tour.start()->schedule.departure, 0, 0, Timing{0, 0, 0, 0}, 0},
        [&](const auto& acc, const auto& indexedActivity) {
          auto [act, activityIndex] = indexedActivity;

          // get activity type
          auto type = getActivityType(*act, activityIndex);
          bool isBreak = type == "break";

          // timings
          auto driving =
            problem.transport->duration(actor.vehicle->profile, acc.location, act->detail.location, acc.departure);
          auto arrival = acc.departure + driving;
          auto start = std::max(act->schedule.arrival, act->detail.time.start);
          auto waiting = start - act->schedule.arrival;
          auto serving = problem.activity->duration(actor, *act, act->schedule.arrival);
          auto departure = start + serving;

          bool isSameLocation = acc.location == act->detail.location;

          // TODO initialize stop
          if (tour.stops.empty() || isSameLocation)
            tour.stops.push_back(Stop{coordIndex.find(act->detail.location),
                                      Schedule{dateConverter(arrival), dateConverter(departure)},
                                      {},
                                      {}});

          /**
           struct Stop final {
            std::vector<double> location;
            Schedule time;
            std::vector<int> load;
            std::vector<Activity> activities;
          };

           struct Activity final {
            std::string jobId;
            std::string type;
            std::optional<std::vector<double>> location;
            std::optional<Interval> time;
          };
           */

          tour.stops.back().time.departure = departure;
          tour.stops.back().load = {};  // TODO
                                        //          tour.stops.back().activities.push_back(Activity{
                                        //            "jobId",
                                        //            type,
                                        //            isSameLocation ? {} : std::make_optional<>(act->detail.location)
          //            isSameLocation ? {} : std::make_optional<Interval>(Interval{arrival, departure})
          //          });

          return Leg{
            act->detail.location,
            act->schedule.departure,
            acc.distance +
              problem.transport->distance(actor.vehicle->profile, acc.location, act->detail.location, acc.departure),
            acc.duration + departure - acc.departure,
            Timing{static_cast<int>(driving),
                   static_cast<int>(acc.timing.serving + (isBreak ? 0 : serving)),
                   static_cast<int>(waiting),
                   static_cast<int>(acc.timing.breakTime + (isBreak ? serving : 0))},
            problem.transport->cost(actor, acc.location, act->detail.location, acc.departure)};
        });

      tour.statistic = Statistic{leg.cost, static_cast<int>(leg.distance), static_cast<int>(leg.duration), leg.timing};
      solution.tours.push_back(tour);

      return Statistic{acc.cost + leg.cost,
                       static_cast<int>(acc.distance + leg.distance),
                       static_cast<int>(acc.duration + leg.duration),
                       Timing{acc.times.driving + leg.timing.driving,
                              acc.times.serving + leg.timing.serving,
                              acc.times.waiting + leg.timing.waiting,
                              acc.times.breakTime + leg.timing.breakTime}};
    });

  return std::move(solution);
}

// endregion
}

struct dump_solution_as_here_json final {
  std::shared_ptr<const models::Problem> problem;
  void operator()(std::ostream& out, const models::EstimatedSolution& es) const {
    nlohmann::json json;
    detail::here::to_json(json, detail::here::createSolution(*problem, es));
    out << json.dump(4);
  }
};
}