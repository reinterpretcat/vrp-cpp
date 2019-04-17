#pragma once

#include "streams/in/extensions/JsonHelpers.hpp"

#include <nlohmann/json.hpp>
#include <optional>

namespace vrp::streams::in::detail::here {

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

// endregion
}