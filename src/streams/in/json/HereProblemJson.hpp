#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/ConditionalJob.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/Problem.hpp"
#include "models/costs/MatrixTransportCosts.hpp"
#include "streams/in/extensions/JsonHelpers.hpp"
#include "utils/Date.hpp"

#include <istream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <range/v3/utility/variant.hpp>
#include <tuple>
#include <unordered_map>

namespace vrp::streams::in {

namespace detail::here {

// region Plan

enum class RelationType { Sequence, Tour, Flexible };

struct Relation final {
  RelationType type;
  std::vector<std::string> jobs;
  std::string vehicleId;
};


struct JobPlace final {
  std::optional<std::vector<std::vector<std::string>>> times;
  std::vector<double> location;
  double duration;
};

struct JobPlaces final {
  std::optional<JobPlace> pickup;
  std::optional<JobPlace> delivery;
};

struct Job final {
  std::string id;
  JobPlaces places;
  std::vector<int> demand;
  std::optional<std::vector<std::string>> skills;
};

struct Plan final {
  std::vector<Job> jobs;
  std::optional<std::vector<Relation>> relations;
};

inline void
from_json(const nlohmann::json& j, RelationType& r) {
  static const std::map<std::string, RelationType> types = {
    {"sequence", RelationType::Sequence},
    {"tour", RelationType::Tour},
    {"flexible", RelationType::Flexible},
  };

  auto type = j.get<std::string>();
  auto value = types.find(type);
  if (value == types.cend()) throw std::invalid_argument("Unknown relation type: '" + type + "'");
  r = value->second;
}

inline void
from_json(const nlohmann::json& j, Relation& r) {
  j.at("type").get_to(r.type);
  j.at("jobs").get_to(r.jobs);
  j.at("vehicleId").get_to(r.vehicleId);
}


inline void
from_json(const nlohmann::json& j, JobPlace& job) {
  readOptional(j, "times", job.times);
  j.at("location").get_to(job.location);
  j.at("duration").get_to(job.duration);
}

inline void
from_json(const nlohmann::json& j, JobPlaces& job) {
  readOptional(j, "pickup", job.pickup);
  readOptional(j, "delivery", job.delivery);
}

inline void
from_json(const nlohmann::json& j, Job& job) {
  j.at("id").get_to(job.id);
  j.at("places").get_to(job.places);
  j.at("demand").get_to(job.demand);

  readOptional(j, "skills", job.skills);
}


// endregion

// region Fleet

struct VehicleCosts final {
  std::optional<double> fixed;
  double distance;
  double time;
};

struct VehiclePlace final {
  std::string time;
  std::vector<double> location;
};

struct VehiclePlaces final {
  VehiclePlace start;
  std::optional<VehiclePlace> end;
};

struct VehicleLimits final {
  std::optional<double> maxDistance;
  std::optional<double> shiftTime;
};

struct VehicleBreak final {
  std::vector<std::vector<std::string>> times;
  double duration;
  std::optional<std::vector<double>> location;
};


inline void
from_json(const nlohmann::json& j, VehicleCosts& v) {
  readOptional(j, "fixed", v.fixed);
  j.at("distance").get_to(v.distance);
  j.at("time").get_to(v.time);
}

inline void
from_json(const nlohmann::json& j, VehiclePlace& v) {
  j.at("time").get_to(v.time);
  j.at("location").get_to(v.location);
}

inline void
from_json(const nlohmann::json& j, VehiclePlaces& v) {
  j.at("start").get_to(v.start);
  readOptional(j, "end", v.end);
}

inline void
from_json(const nlohmann::json& j, VehicleLimits& v) {
  readOptional(j, "maxDistance", v.maxDistance);
  readOptional(j, "shiftTime", v.shiftTime);
}

inline void
from_json(const nlohmann::json& j, VehicleBreak& v) {
  j.at("times").get_to(v.times);
  j.at("duration").get_to(v.duration);
  readOptional(j, "location", v.location);
}

struct VehicleType final {
  std::string id;
  std::string profile;
  VehicleCosts costs;
  VehiclePlaces places;
  std::vector<int> capacity;

  std::optional<std::vector<std::string>> skills;
  std::optional<VehicleLimits> limits;
  std::optional<VehicleBreak> vehicleBreak;

  int amount;
};

struct Fleet final {
  std::vector<VehicleType> types;
};

inline void
from_json(const nlohmann::json& j, VehicleType& v) {
  j.at("id").get_to(v.id);
  j.at("profile").get_to(v.profile);
  j.at("costs").get_to(v.costs);
  j.at("places").get_to(v.places);
  j.at("capacity").get_to(v.capacity);

  readOptional(j, "skills", v.skills);
  readOptional(j, "limits", v.limits);
  readOptional(j, "break", v.vehicleBreak);

  j.at("amount").get_to(v.amount);
}

// endregion

// region Routing

struct Matrix final {
  std::string profile;
  std::vector<double> distances;
  std::vector<double> durations;
};

inline void
from_json(const nlohmann::json& j, Matrix& m) {
  j.at("profile").get_to(m.profile);
  j.at("distances").get_to(m.distances);
  j.at("durations").get_to(m.durations);
}

// endregion

// region Root

struct Problem final {
  std::string id;
  Plan plan;
  Fleet fleet;
  std::vector<Matrix> matrices;
};

inline void
from_json(const nlohmann::json& j, Fleet& f) {
  j.at("types").get_to(f.types);
}

inline void
from_json(const nlohmann::json& j, Plan& p) {
  j.at("jobs").get_to(p.jobs);
  readOptional(j, "relations", p.relations);
}

inline void
from_json(const nlohmann::json& j, Problem& p) {
  j.at("id").get_to(p.id);
  j.at("plan").get_to(p.plan);
  j.at("fleet").get_to(p.fleet);
  j.at("matrices").get_to(p.matrices);
}

struct BreakConstraint final : public vrp::algorithms::construction::HardActivityConstraint {
  // TODO decorate ConditionalJob + ensure break first

  BreakConstraint() :
    conditionalJob_(
      [](const vrp::algorithms::construction::InsertionSolutionContext& ctx, const models::problem::Job& job) {
        return models::problem::analyze_job<bool>(
          job,
          [&ctx](const std::shared_ptr<const models::problem::Service>& service) {
            // mark service as ignored only if it has break type and vehicle id is not present in routes
            if (isNotBreak(service)) return true;

            const auto& vehicleId = std::any_cast<std::string>(service->dimens.at("vehicleId"));
            return ranges::find_if(ctx.routes, [&vehicleId](const auto& iCtx) {
                     // TODO check arrival time at last activity to avoid assigning break as last
                     return std::any_cast<std::string>(iCtx.route->actor->vehicle->dimens.at("id")) == vehicleId;
                   }) != ctx.routes.end();
          },
          [](const std::shared_ptr<const models::problem::Sequence>& sequence) { return true; });
      }) {}

  ranges::any_view<int> stateKeys() const override { return ranges::view::empty<int>(); }

  void accept(vrp::algorithms::construction::InsertionSolutionContext& ctx) const override {
    conditionalJob_.accept(ctx);
  }

  void accept(vrp::algorithms::construction::InsertionRouteContext&) const override {}

  vrp::algorithms::construction::HardActivityConstraint::Result hard(
    const vrp::algorithms::construction::InsertionRouteContext& routeCtx,
    const vrp::algorithms::construction::InsertionActivityContext& actCtx) const override {
    using namespace vrp::algorithms::construction;
    // TODO check that break is not assigned as last?
    return isNotBreak(actCtx.target->service.value()) || actCtx.prev->service.has_value() ? success() : stop(4);
  }

private:
  static bool isNotBreak(const std::shared_ptr<const models::problem::Service>& service) {
    auto type = service->dimens.find("type");
    if (type == service->dimens.end() || std::any_cast<std::string>(type->second) != "break") return true;

    return false;
  }

  vrp::algorithms::construction::ConditionalJob conditionalJob_;
};

// endregion
}

/// Represents coordinate index.
struct CoordIndex final {
  void add(const std::vector<double>& location) {
    assert(location.size() == 2);
    auto value = std::make_pair(location[0], location[1]);
    if (coordToIndex.find(value) == coordToIndex.end()) {
      auto index = coordToIndex.size();
      coordToIndex[value] = index;
      indexToCoord[index] = value;
    }
  }

  std::vector<double> find(size_t index) const {
    auto pair = indexToCoord.at(index);
    return {pair.first, pair.second};
  }

  size_t find(const std::vector<double>& location) const {
    assert(location.size() == 2);
    return coordToIndex.at(std::make_pair(location[0], location[1]));
  }

private:
  struct coord_less final {
    bool operator()(const std::pair<double, double>& lhs, const std::pair<double, double>& rhs) const {
      return std::tie(lhs.first, lhs.second) < std::tie(rhs.first, rhs.second);
    }
  };

  std::map<std::pair<double, double>, size_t, coord_less> coordToIndex;
  std::map<size_t, std::pair<double, double>> indexToCoord;
};

/// Parses HERE VRP problem definition from json.
struct read_here_json_type {
public:
  std::shared_ptr<models::Problem> operator()(std::istream& input) const {
    using namespace vrp::algorithms::construction;
    using namespace vrp::models;

    nlohmann::json j;
    input >> j;

    auto problem = j.get<detail::here::Problem>();

    auto coordIndex = createCoordIndex(problem);

    auto transport = transportCosts(problem);
    auto activity = std::make_shared<costs::ActivityCosts>();

    auto fleet = readFleet(problem, coordIndex);
    auto jobs = readJobs(problem, coordIndex, *transport, *fleet);

    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, transport, activity))
      .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>())
      .addHardActivity(std::make_shared<detail::here::BreakConstraint>());

    auto extras = std::make_shared<std::map<std::string, std::any>>();
    extras->insert(std::make_pair("coordIndex", std::move(coordIndex)));
    extras->insert(std::make_pair("problemId", problem.id));

    return std::make_shared<models::Problem>(
      models::Problem{fleet,
                      jobs,
                      constraint,
                      std::make_shared<algorithms::objectives::penalize_unassigned_jobs<10000>>(),
                      activity,
                      transport,
                      extras});
  }

private:
  /// Creates coordinate index to match routing data.
  CoordIndex createCoordIndex(const detail::here::Problem& problem) const {
    auto index = CoordIndex{};
    ranges::for_each(problem.plan.jobs, [&](const auto& job) {
      if (job.places.pickup) index.add(job.places.pickup.value().location);
      if (job.places.delivery) index.add(job.places.delivery.value().location);
    });

    ranges::for_each(problem.fleet.types, [&](const auto& vehicle) {
      index.add(vehicle.places.start.location);
      if (vehicle.places.end) index.add(vehicle.places.end.value().location);
      if (vehicle.vehicleBreak && vehicle.vehicleBreak.value().location)
        index.add(vehicle.vehicleBreak.value().location.value());
    });

    return std::move(index);
  }

  std::shared_ptr<models::problem::Fleet> readFleet(const detail::here::Problem& problem,
                                                    const CoordIndex& coordIndex) const {
    using namespace vrp::algorithms::construction;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;
    using namespace vrp::utils;

    auto dateParser = parse_date_from_rc3339{};

    auto fleet = std::make_shared<Fleet>();
    ranges::for_each(problem.fleet.types, [&](const auto& vehicle) {
      assert(vehicle.capacity.size() == 1);

      auto details = std::vector<Vehicle::Detail>{
        Vehicle::Detail{coordIndex.find(vehicle.places.start.location),
                        vehicle.places.end ? std::make_optional(coordIndex.find(vehicle.places.end.value().location))
                                           : std::optional<models::common::Location>{},
                        TimeWindow{static_cast<double>(dateParser(vehicle.places.start.time)),
                                   vehicle.places.end ? static_cast<double>(dateParser(vehicle.places.end.value().time))
                                                      : std::numeric_limits<double>::max()}}};

      ranges::for_each(ranges::view::closed_indices(1, vehicle.amount), [&](auto index) {
        fleet->add(Vehicle{vehicle.profile,
                           Costs{vehicle.costs.fixed.value_or(0),
                                 vehicle.costs.distance,
                                 vehicle.costs.time,
                                 vehicle.costs.time,
                                 vehicle.costs.time},

                           Dimensions{{"typeId", vehicle.id},
                                      {"id", vehicle.id + "_" + std::to_string(index)},
                                      {VehicleActivitySize<int>::DimKeyCapacity, vehicle.capacity.front()}},

                           details});
      });
    });

    fleet->add(Driver{Costs{0, 0, 0, 0, 0}, Dimensions{{"id", "driver"}}});

    return fleet;
  }

  std::shared_ptr<models::problem::Jobs> readJobs(const detail::here::Problem& problem,
                                                  const CoordIndex& coordIndex,
                                                  const models::costs::TransportCosts& transport,
                                                  const models::problem::Fleet& fleet) const {
    return std::make_shared<models::problem::Jobs>(models::problem::Jobs{
      transport,
      fleet,
      ranges::view::concat(readRequiredJobs(problem, coordIndex), readConditionalJobs(problem, coordIndex)),
    });
  }

  ranges::any_view<models::problem::Job> readRequiredJobs(const detail::here::Problem& problem,
                                                          const CoordIndex& coordIndex) const {
    using namespace ranges;
    using namespace algorithms::construction;
    using namespace models::common;
    using namespace models::problem;

    auto dateParser = vrp::utils::parse_date_from_rc3339{};

    return view::for_each(problem.plan.jobs, [&, dateParser](const auto& job) {
      assert(job.places.pickup || job.places.delivery);

      static auto createDemand = [](const auto& job, bool isPickup, bool isFixed) {
        assert(job.demand.size() == 1);
        auto value = job.demand.front();
        return VehicleActivitySize<int>::Demand{{isFixed && isPickup ? value : 0, !isFixed && isPickup ? value : 0},
                                                {isFixed && !isPickup ? value : 0, !isFixed && !isPickup ? value : 0}};
      };

      auto createService = [&, dateParser](const std::string& id, const auto& place, const auto& demand) {
        auto times = place.times.has_value()  //
          ? ranges::accumulate(place.times.value(),
                               std::vector<TimeWindow>{},
                               [&](auto& acc, const auto& time) {
                                 acc.push_back({static_cast<double>(dateParser(time.at(0))),
                                                static_cast<double>(dateParser(time.at(1)))});
                                 return std::move(acc);
                               })
          : std::vector<TimeWindow>{TimeWindow{0, std::numeric_limits<double>::max()}};
        return std::make_shared<Service>(
          Service{{Service::Detail{coordIndex.find(place.location), place.duration, std::move(times)}},
                  Dimensions{{"id", id}, {VehicleActivitySize<int>::DimKeyDemand, demand}}});
      };

      // shipment
      if (job.places.pickup && job.places.delivery) {
        return yield(Job{ranges::emplaced_index<1>,
                         std::make_shared<Sequence>(Sequence{
                           {createService(job.id, job.places.pickup.value(), createDemand(job, true, false)),
                            createService(job.id, job.places.delivery.value(), createDemand(job, false, false))},
                           Dimensions{{"id", job.id}}})});
        // pickup
      } else if (job.places.pickup) {
        return yield(Job{ranges::emplaced_index<0>,
                         createService(job.id, job.places.pickup.value(), createDemand(job, true, true))});
      }

      // delivery
      return yield(Job{ranges::emplaced_index<0>,
                       createService(job.id, job.places.delivery.value(), createDemand(job, false, true))});
    });
  }


  ranges::any_view<models::problem::Job> readConditionalJobs(const detail::here::Problem& problem,
                                                             const CoordIndex& coordIndex) const {
    using namespace ranges;
    using namespace algorithms::construction;
    using namespace models::common;
    using namespace models::problem;

    auto dateParser = vrp::utils::parse_date_from_rc3339{};

    // convert vehicle breaks to conditional jobs
    return view::for_each(
      problem.fleet.types | ranges::view::filter([](const auto& v) { return v.vehicleBreak.has_value(); }),
      [&, dateParser](const auto& vehicle) {
        const auto& model = vehicle.vehicleBreak.value();
        const auto id = vehicle.id;
        const auto detail = Service::Detail{
          model.location ? std::make_optional<Location>(Location{coordIndex.find(model.location.value())})
                         : std::make_optional<Location>(),
          model.duration,
          ranges::accumulate(model.times, std::vector<TimeWindow>{}, [&](auto& acc, const auto& time) {
            acc.push_back({static_cast<double>(dateParser(time.at(0))), static_cast<double>(dateParser(time.at(1)))});
            return std::move(acc);
          })};

        return view::for_each(ranges::view::closed_indices(1, vehicle.amount), [=](auto index) {
          return yield(
            Job{ranges::emplaced_index<0>,
                std::make_shared<Service>(Service{{detail},
                                                  Dimensions{{"id", std::string("break")},
                                                             {"type", std::string("break")},
                                                             {"vehicleId", id + "_" + std::to_string(index)}}})});
        });
      });
  }

  std::shared_ptr<models::costs::MatrixTransportCosts> transportCosts(const detail::here::Problem& problem) const {
    using namespace vrp::models::costs;

    auto durations = MatrixTransportCosts::DurationProfiles{};
    auto distances = MatrixTransportCosts::DistanceProfiles{};
    ranges::for_each(problem.matrices, [&](const auto& matrix) {
      // TODO check that each profile is defined only once.
      durations[matrix.profile] = std::move(matrix.durations);
      distances[matrix.profile] = std::move(matrix.distances);
    });

    return std::make_shared<MatrixTransportCosts>(MatrixTransportCosts{std::move(durations), std::move(distances)});
  }
};
}