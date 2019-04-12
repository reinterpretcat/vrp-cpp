#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/Problem.hpp"
#include "models/costs/MatrixTransportCosts.hpp"
#include "streams/in/extensions/JsonHelpers.hpp"

#include <istream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <range/v3/utility/variant.hpp>

namespace vrp::streams::in {

namespace detail::rich {
// region Driver

struct DriverType {
  std::string id;
  int amount;
};

void
from_json(const nlohmann::json& j, DriverType& v) {
  j.at("id").get_to(v.id);
  j.at("amount").get_to(v.amount);
}

// endregion

// region Vehicle

struct VehicleBreak {
  double time;
  double duration;
  std::uint64_t location;
};

struct VehicleCapabilities {
  std::vector<int> capacity;
  std::vector<std::string> skills;
};

struct VehicleDetail {
  Location start;
  std::optional<Location> end;
  TimeWindow time;
};

struct VehicleCosts {
  double fixed;
  double distance;
  double driving;
  double waiting;
  double serving;
};

struct VehicleLimits {
  double maxDistance;
  double maxTime;
};

struct VehicleType {
  std::string id;
  std::string profile;
  int amount;

  std::vector<VehicleDetail> details;
  VehicleCosts costs;

  std::optional<VehicleLimits> limits;
  std::optional<VehicleCapabilities> capabilities;
  std::optional<VehicleBreak> breaks;
};

void
from_json(const nlohmann::json& j, VehicleDetail& v) {
  j.at("start").get_to(v.start);
  j.at("end").get_to(v.end);
  j.at("time").get_to(v.time);
}

void
from_json(const nlohmann::json& j, VehicleBreak& v) {
  j.at("time").get_to(v.time);
  j.at("duration").get_to(v.duration);
  j.at("location").get_to(v.location);
}

void
from_json(const nlohmann::json& j, VehicleCapabilities& v) {
  j.at("capacity").get_to(v.capacity);
  j.at("skills").get_to(v.skills);
}

void
from_json(const nlohmann::json& j, VehicleLimits& v) {
  j.at("max_distance").get_to(v.maxDistance);
  j.at("max_time").get_to(v.maxTime);
}

void
from_json(const nlohmann::json& j, VehicleCosts& v) {
  j.at("fixed").get_to(v.fixed);
  j.at("distance").get_to(v.distance);
  j.at("driving").get_to(v.driving);
  j.at("waiting").get_to(v.waiting);
  j.at("serving").get_to(v.serving);
}

void
from_json(const nlohmann::json& j, VehicleType& v) {
  j.at("id").get_to(v.id);
  j.at("profile").get_to(v.profile);
  j.at("details").get_to(v.details);

  j.at("costs").get_to(v.costs);

  j.at("amount").get_to(v.amount);

  readOptional(j, "capabilities", v.capabilities);
  readOptional(j, "limits", v.limits);
  readOptional(j, "breaks", v.breaks);
}

// endregion

// region Job

struct Demand final {
  std::optional<std::vector<int>> pickup;
  std::optional<std::vector<int>> delivery;
};

struct Demands final {
  std::optional<Demand> fixed;
  std::optional<Demand> dynamic;
};

struct ServiceDetail {
  std::optional<std::uint64_t> location;
  Timestamp duration;
  std::vector<TimeWindow> times;
};

struct ServiceRequirements {
  Demands demands;
  std::vector<std::string> skills;
};

struct Service {
  std::optional<ServiceRequirements> requirements;
  std::vector<ServiceDetail> details;
};

struct Sequence {
  std::vector<Service> services;
};

struct Job {
  std::string id;
  ranges::variant<Service, Sequence> variant;
};

void
from_json(const nlohmann::json& j, Demand& d) {
  readOptional(j, "pickup", d.pickup);
  readOptional(j, "delivery", d.delivery);
}

void
from_json(const nlohmann::json& j, Demands& d) {
  readOptional(j, "fixed", d.fixed);
  readOptional(j, "dynamic", d.dynamic);
}

void
from_json(const nlohmann::json& j, ServiceDetail& s) {
  j.at("duration").get_to(s.duration);
  j.at("times").get_to(s.times);
  readOptional(j, "location", s.location);
}

void
from_json(const nlohmann::json& j, Service& s) {
  j.at("details").get_to(s.details);
  readOptional(j, "requirements", s.requirements);
}

void
from_json(const nlohmann::json& j, Sequence& s) {
  j.at("services").get_to(s.services);
}

void
from_json(const nlohmann::json& j, ServiceRequirements& s) {
  j.at("skills").get_to(s.skills);
  j.at("demands").get_to(s.demands);
}

void
from_json(const nlohmann::json& j, Job& job) {
  j.at("id").get_to(job.id);

  auto type = j.at("type").get<std::string>();
  if (type == "service") {
    auto service = Service{};
    j.get_to(service);
    job.variant = ranges::variant<Service, Sequence>{ranges::emplaced_index<0>, service};
  } else if (type == "sequence") {
    auto sequence = Sequence{};
    j.get_to(sequence);
    job.variant = ranges::variant<Service, Sequence>{ranges::emplaced_index<1>, sequence};
  } else {
    throw std::invalid_argument("Unknown job type: '" + type + "'");
  }
}

// endregion

// region Routing

struct Matrix {
  std::string profile;
  std::vector<double> distances;
  std::vector<double> durations;
};

struct Routing {
  std::vector<Matrix> matrices;
};

void
from_json(const nlohmann::json& j, Matrix& m) {
  j.at("profile").get_to(m.profile);
  j.at("distances").get_to(m.distances);
  j.at("durations").get_to(m.durations);
}

void
from_json(const nlohmann::json& j, Routing& r) {
  j.at("matrices").get_to(r.matrices);
}

// endregion

// region Route

enum class ActivityOrder { Any, Sequence, Strict };

struct Route final {
  std::string vehicleId;
  ActivityOrder order;
  std::vector<std::string> jobs;
};

void
from_json(const nlohmann::json& j, ActivityOrder& r) {
  static const std::map<std::string, ActivityOrder> types = {
    {"any", ActivityOrder::Any},
    {"sequence", ActivityOrder::Sequence},
    {"strict", ActivityOrder::Strict},
  };

  auto type = j.get<std::string>();
  auto value = types.find(type);
  if (value == types.cend()) throw std::invalid_argument("Unknown route type: '" + type + "'");
  r = value->second;
}

void
from_json(const nlohmann::json& j, Route& r) {
  j.at("vehicleId").get_to(r.vehicleId);
  j.at("order").get_to(r.order);
  j.at("jobs").get_to(r.jobs);
}

// endregion

// region Root

struct Fleet {
  std::vector<DriverType> drivers;
  std::vector<VehicleType> vehicles;
};

struct Plan final {
  std::vector<Job> jobs;
  std::vector<Route> routes;
};

struct Problem {
  std::string id;
  Fleet fleet;
  Plan plan;
  Routing routing;
};

void
from_json(const nlohmann::json& j, Fleet& f) {
  j.at("drivers").get_to(f.drivers);
  j.at("vehicles").get_to(f.vehicles);
}

void
from_json(const nlohmann::json& j, Plan& p) {
  j.at("jobs").get_to(p.jobs);
  j.at("routes").get_to(p.routes);
}

void
from_json(const nlohmann::json& j, Problem& p) {
  j.at("id").get_to(p.id);
  j.at("fleet").get_to(p.fleet);
  j.at("plan").get_to(p.plan);
  j.at("routing").get_to(p.routing);
}

// endregion
}

/// Parses rich VRP from json.
struct read_rich_json_type {
  std::shared_ptr<models::Problem> operator()(std::istream& input) const {
    using namespace vrp::algorithms::construction;

    nlohmann::json j;
    input >> j;

    auto problem = j.get<detail::rich::Problem>();

    auto transport = transportCosts(problem);
    auto activity = std::make_shared<models::costs::ActivityCosts>();

    auto fleet = readFleet(problem);
    auto jobs = readJobs(problem, *transport, *fleet);

    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, transport, activity))
      .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>());

    return std::make_shared<models::Problem>(
      models::Problem{fleet,
                      jobs,
                      constraint,
                      std::make_shared<algorithms::objectives::penalize_unassigned_jobs<10000>>(),
                      activity,
                      transport,
                      {}});
  }

private:
  std::shared_ptr<models::problem::Fleet> readFleet(const detail::rich::Problem& problem) const {
    using namespace vrp::algorithms::construction;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;

    assert(problem.fleet.drivers.size() == 1);

    auto fleet = std::make_shared<Fleet>();
    ranges::for_each(problem.fleet.vehicles, [&](const auto& vehicle) {
      ranges::for_each(ranges::view::closed_indices(1, vehicle.amount), [&](auto index) {
        assert(vehicle.capabilities.has_value());
        assert(vehicle.capabilities.value().capacity.size() == 1);

        fleet->add(
          Vehicle{getProfile(vehicle.profile),

                  Costs{vehicle.costs.fixed,
                        vehicle.costs.distance,
                        vehicle.costs.driving,
                        vehicle.costs.waiting,
                        vehicle.costs.serving},

                  Dimensions{{"id", vehicle.id + "_" + std::to_string(index)},
                             {VehicleActivitySize<int>::DimKeyCapacity, vehicle.capabilities.value().capacity.front()}},

                  ranges::accumulate(vehicle.details, std::vector<Vehicle::Detail>{}, [](auto& acc, const auto detail) {
                    acc.push_back(Vehicle::Detail{detail.start, detail.end, {detail.time.start, detail.time.end}});
                    return std::move(acc);
                  })});
      });
    });

    ranges::for_each(problem.fleet.drivers, [&](const auto& driver) {
      ranges::for_each(ranges::view::closed_indices(1, driver.amount), [&](auto index) {
        using namespace vrp::models::common;
        using namespace vrp::models::problem;

        // TODO implement driver costs
        fleet->add(Driver{Costs{0, 0, 0, 0, 0}, Dimensions{{"id", driver.id + "_" + std::to_string(index)}}});
      });
    });

    return fleet;
  }

  std::shared_ptr<models::problem::Jobs> readJobs(const detail::rich::Problem& problem,
                                                  const models::costs::TransportCosts& transport,
                                                  const models::problem::Fleet& fleet) const {
    using namespace ranges;
    using namespace algorithms::construction;
    using namespace models::common;
    using namespace models::problem;

    auto jobs = view::for_each(problem.plan.jobs, [](const auto& job) {
      static auto ensureDemand = [](const auto& demand) {
        auto newDemand = demand.value_or(detail::rich::Demand{{{0}}, {{0}}});

        newDemand.pickup = newDemand.pickup.value_or(std::vector<int>{0});
        newDemand.delivery = newDemand.delivery.value_or(std::vector<int>{0});

        if (newDemand.pickup.value().empty()) newDemand.pickup.value().push_back(0);
        if (newDemand.delivery.value().empty()) newDemand.delivery.value().push_back(0);

        assert(newDemand.pickup.value().size() == 1 && newDemand.delivery.value().size() == 1);

        return std::move(newDemand);
      };

      static auto createService = [](const auto& s, const std::string& id) {
        assert(s.requirements.has_value());
        auto fixed = ensureDemand(s.requirements.value().demands.fixed);
        auto dynamic = ensureDemand(s.requirements.value().demands.dynamic);

        return std::make_shared<Service>(Service{
          // details
          ranges::accumulate(s.details,
                             std::vector<Service::Detail>{},
                             [](auto& acc, const auto detail) {
                               auto times = ranges::accumulate(
                                 detail.times, std::vector<TimeWindow>{}, [](auto& acc, const auto& time) {
                                   acc.push_back({time.start, time.end});
                                   return std::move(acc);
                                 });

                               acc.push_back(Service::Detail{detail.location, detail.duration, std::move(times)});
                               return std::move(acc);
                             }),
          // demand
          Dimensions{
            {"id", id},
            {VehicleActivitySize<int>::DimKeyDemand,
             VehicleActivitySize<int>::Demand{{fixed.pickup.value().front(), dynamic.pickup.value().front()},
                                              {fixed.delivery.value().front(), dynamic.delivery.value().front()}}}}});
      };

      auto result = job.variant.visit(ranges::overload(  //
        [&](const detail::rich::Service& s) -> models::problem::Job {
          return Job{ranges::emplaced_index<0>, createService(s, job.id)};
        },
        [&](const detail::rich::Sequence& s) -> models::problem::Job {
          // TODO use build_sequence as it adds specific dimens
          return models::problem::Job{
            ranges::emplaced_index<1>,
            std::make_shared<Sequence>(  //
              Sequence{ranges::accumulate(view::zip(s.services, view::iota(1)),
                                          std::vector<std::shared_ptr<const Service>>{},
                                          [&](auto& acc, const auto& pair) {
                                            const auto& [srv, index] = pair;

                                            acc.push_back(createService(srv, job.id + "_" + std::to_string(index)));
                                            return std::move(acc);
                                          }),
                       Dimensions{{"id", job.id}}})};
        }));
      return yield(result.index() == 0 ? ranges::get<0>(result) : ranges::get<1>(result));
    });


    return std::make_shared<models::problem::Jobs>(models::problem::Jobs{transport, fleet, jobs});
  }

  std::shared_ptr<models::costs::MatrixTransportCosts> transportCosts(const detail::rich::Problem& problem) const {
    using namespace vrp::models::costs;

    auto durations = MatrixTransportCosts::DurationProfiles{};
    auto distances = MatrixTransportCosts::DistanceProfiles{};
    durations.resize(2);
    distances.resize(2);
    ranges::for_each(problem.routing.matrices, [&](const auto& matrix) {
      // TODO check that each profile is defined only once.
      auto profile = getProfile(matrix.profile);
      durations[profile] = std::move(matrix.durations);
      distances[profile] = std::move(matrix.distances);
    });

    return std::make_shared<MatrixTransportCosts>(MatrixTransportCosts{std::move(durations), std::move(distances)});
  }

  models::common::Profile getProfile(const std::string& value) const {
    if (value == "car") return 0;
    if (value == "truck") return 1;

    throw std::invalid_argument(std::string("Unknown routing profile: ") + value);
  }
};
}