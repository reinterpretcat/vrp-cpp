#pragma once

#include "streams/in/extensions/JsonHelpers.hpp"

#include <nlohmann/json.hpp>
#include <optional>

namespace vrp::streams::in::detail::rich {

// region Common

struct RichLocation final {
  double latitude;
  double longitude;
};

struct RichTime final {
  std::string start;
  std::string end;
};

void
from_json(const nlohmann::json& j, RichLocation& loc) {
  j.at("lat").get_to(loc.latitude);
  j.at("lon").get_to(loc.longitude);
}

void
from_json(const nlohmann::json& j, RichTime& time) {
  j.at("start").get_to(time.start);
  j.at("end").get_to(time.end);
}

// endregion

// region Driver

struct DriverCosts {
  double fixed;
  double distance;
  double driving;
  double waiting;
  double serving;
};

struct DriverLimits {
  double maxTime;
};

struct DriverBreak {
  RichTime time;
  double duration;
  std::optional<RichLocation> location;
};

struct DriverCapabilities {
  std::optional<std::vector<std::string>> skills;
  std::vector<std::string> profiles;
  std::optional<std::vector<std::string>> vehicles;
};

struct DriverAvailability {
  std::optional<RichTime> time;
  std::optional<DriverBreak> breakTime;
};

struct DriverType {
  std::string id;
  int amount;

  DriverCosts costs;
  DriverCapabilities capabilities;
  std::vector<DriverAvailability> availability;

  std::optional<DriverLimits> limits;
};

void
from_json(const nlohmann::json& j, DriverLimits& d) {
  j.at("maxTime").get_to(d.maxTime);
}

void
from_json(const nlohmann::json& j, DriverCosts& d) {
  j.at("fixed").get_to(d.fixed);
  j.at("distance").get_to(d.distance);
  j.at("driving").get_to(d.driving);
  j.at("waiting").get_to(d.waiting);
  j.at("serving").get_to(d.serving);
}

void
from_json(const nlohmann::json& j, DriverBreak& d) {
  j.at("time").get_to(d.time);
  j.at("duration").get_to(d.duration);
  j.at("location").get_to(d.location);
}

void
from_json(const nlohmann::json& j, DriverAvailability& d) {
  j.at("time").get_to(d.time);
  readOptional(j, "break", d.breakTime);
}

void
from_json(const nlohmann::json& j, DriverCapabilities& d) {
  readOptional(j, "skills", d.skills);
  j.at("profiles").get_to(d.profiles);
  readOptional(j, "vehicles", d.vehicles);
}

void
from_json(const nlohmann::json& j, DriverType& d) {
  j.at("id").get_to(d.id);

  j.at("costs").get_to(d.costs);
  j.at("amount").get_to(d.amount);

  j.at("availability").get_to(d.availability);
  j.at("capabilities").get_to(d.capabilities);

  readOptional(j, "limits", d.limits);
}

// endregion

// region Vehicle

struct VehicleCapabilities {
  std::vector<int> capacities;
  std::optional<std::vector<std::string>> facilities;
};

struct VehicleLocations {
  RichLocation start;
  std::optional<RichLocation> end;
};

struct VehicleAvailability {
  VehicleLocations location;
  std::optional<RichTime> time;
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
};

struct VehicleType {
  std::string id;
  std::string profile;
  int amount;

  std::vector<VehicleAvailability> availability;
  VehicleCosts costs;
  std::optional<VehicleCapabilities> capabilities;
  std::optional<VehicleLimits> limits;
};

void
from_json(const nlohmann::json& j, VehicleLocations& loc) {
  j.at("start").get_to(loc.start);
  readOptional(j, "end", loc.end);
}

void
from_json(const nlohmann::json& j, VehicleAvailability& v) {
  j.at("location").get_to(v.location);
  j.at("time").get_to(v.time);
}

void
from_json(const nlohmann::json& j, VehicleCapabilities& v) {
  readOptional(j, "capacities", v.capacities);
  readOptional(j, "facilities", v.facilities);
}

void
from_json(const nlohmann::json& j, VehicleLimits& v) {
  j.at("maxDistance").get_to(v.maxDistance);
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

  j.at("costs").get_to(v.costs);
  j.at("amount").get_to(v.amount);

  j.at("availability").get_to(v.availability);
  j.at("capabilities").get_to(v.capabilities);

  readOptional(j, "limits", v.limits);
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
  std::optional<RichLocation> location;
  Timestamp duration;
  std::vector<RichTime> times;
};

struct ServiceRequirements {
  Demands demands;
  std::optional<std::vector<std::string>> skills;
  std::optional<std::vector<std::string>> facilities;
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

  readOptional(j, "routes", p.routes);
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