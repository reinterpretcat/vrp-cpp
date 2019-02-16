#pragma once

#include "models/Problem.hpp"

#include <istream>
#include <nlohmann/json.hpp>
#include <optional>
#include <range/v3/utility/variant.hpp>

namespace nlohmann {
template<typename T>
struct adl_serializer<std::optional<T>> {
  static void to_json(json& j, const std::optional<T>& opt) {
    if (opt)
      j = *opt;
    else
      j = nullptr;
  }

  static void from_json(const json& j, std::optional<T>& opt) {
    if (j.is_null())
      opt = {};
    else
      opt = j.get<T>();
  }
};
}

namespace vrp::streams::in {

namespace detail {

template<typename T>
void
readOptional(const nlohmann::json& j, const std::string& key, T& v) {
  if (j.find(key) != j.end()) j.at(key).get_to(v);
}

// region Common

using Location = std::uint64_t;
using Timestamp = double;

struct TimeWindow {
  Timestamp start;
  Timestamp end;
};

void
from_json(const nlohmann::json& j, TimeWindow& tw) {
  j.at("start").get_to(tw.start);
  j.at("end").get_to(tw.end);
}

// endregion

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
  double distance;
  double time;
  double fixed;
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
  j.at("distance").get_to(v.distance);
  j.at("time").get_to(v.time);
  j.at("fixed").get_to(v.fixed);
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
  ranges::variant<Service, Sequence> job;
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
    job.job = ranges::variant<Service, Sequence>{ranges::emplaced_index<0>, service};
  } else if (type == "sequence") {
    auto sequence = Sequence{};
    j.get_to(sequence);
    job.job = ranges::variant<Service, Sequence>{ranges::emplaced_index<1>, sequence};
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

enum class RouteType { Random, Sequence, Strict };

struct Route final {
  std::string vehicleId;
  RouteType type;
  std::vector<std::string> jobs;
};

void
from_json(const nlohmann::json& j, RouteType& r) {
  static const std::map<std::string, RouteType> types = {
    {"random", RouteType::Random},
    {"sequence", RouteType::Sequence},
    {"strict", RouteType::Strict},
  };

  auto type = j.get<std::string>();
  auto value = types.find(type);
  if (value == types.cend()) throw std::invalid_argument("Unknown route type: '" + type + "'");
  r = value->second;
}

void
from_json(const nlohmann::json& j, Route& r) {
  j.at("vehicleId").get_to(r.vehicleId);
  j.at("type").get_to(r.type);
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
    nlohmann::json j;
    input >> j;

    auto problem = j.get<detail::Problem>();

    // TODO convert detail::problem to models::problem

    return {};
  }
};
}