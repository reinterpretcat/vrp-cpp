#pragma once

#include "streams/in/extensions/JsonHelpers.hpp"

#include <nlohmann/json.hpp>
#include <optional>
#include <range/v3/utility/variant.hpp>

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
  std::optional<std::string> tag;
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

struct MultiJobPlace final {
  std::optional<std::vector<std::vector<std::string>>> times;
  std::vector<double> location;
  double duration;
  std::vector<int> demand;
  std::optional<std::string> tag;
};

struct MultiJobPlaces final {
  std::vector<MultiJobPlace> pickups;
  std::vector<MultiJobPlace> deliveries;
};

struct MultiJob final {
  std::string id;
  MultiJobPlaces places;
  std::optional<std::vector<std::string>> skills;
};

using JobVariant = ranges::variant<Job, MultiJob>;

template<typename Return, typename JobFunc, typename MultiJobFunc>
Return
analyze_variant(const JobVariant& variant, JobFunc&& jobFunc, MultiJobFunc&& multiJobFunc) {
  if (variant.index() == 0) return jobFunc(ranges::get<0>(variant));

  return multiJobFunc(ranges::get<1>(variant));
}

struct Plan final {
  std::vector<JobVariant> jobs;
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
from_json(const nlohmann::json& j, JobPlace& place) {
  readOptional(j, "times", place.times);
  j.at("location").get_to(place.location);
  j.at("duration").get_to(place.duration);
  readOptional(j, "tag", place.tag);
}

inline void
from_json(const nlohmann::json& j, JobPlaces& places) {
  readOptional(j, "pickup", places.pickup);
  readOptional(j, "delivery", places.delivery);
}

inline void
from_json(const nlohmann::json& j, MultiJobPlace& place) {
  readOptional(j, "times", place.times);
  j.at("location").get_to(place.location);
  j.at("duration").get_to(place.duration);
  j.at("demand").get_to(place.demand);
  readOptional(j, "tag", place.tag);
}

inline void
from_json(const nlohmann::json& j, MultiJobPlaces& places) {
  readOptional(j, "pickups", places.pickups);
  readOptional(j, "deliveries", places.deliveries);
}

inline void
from_json(const nlohmann::json& j, JobVariant& variant) {
  if (j.find("demand") != j.end()) {
    auto job = Job{};

    j.at("id").get_to(job.id);
    j.at("places").get_to(job.places);
    j.at("demand").get_to(job.demand);
    readOptional(j, "skills", job.skills);

    variant = JobVariant{ranges::emplaced_index<0>, job};
  } else {
    auto multi = MultiJob{};

    j.at("id").get_to(multi.id);
    j.at("places").get_to(multi.places);
    readOptional(j, "skills", multi.skills);

    variant = JobVariant{ranges::emplaced_index<1>, multi};
  }
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