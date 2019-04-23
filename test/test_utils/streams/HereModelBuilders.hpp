#pragma once

#include <nlohmann/json.hpp>
#include <sstream>

namespace vrp::test::here {

// region Plan
/// Provides the way to create serialized representation of delivery job.
struct build_test_single_job {
  explicit build_test_single_job(std::string type) : type_(type) {}

  build_test_single_job id(nlohmann::json value) {
    content_["id"] = std::move(value);
    return *this;
  }

  build_test_single_job location(double lat, double lon) {
    content_["places"][type_]["location"] = nlohmann::json::array({lat, lon});
    return *this;
  }

  build_test_single_job duration(int duration) {
    content_["places"][type_]["duration"] = duration;
    return *this;
  }

  build_test_single_job times(nlohmann::json times) {
    content_["places"][type_]["times"] = std::move(times);
    return *this;
  }

  build_test_single_job demand(int demand) {
    content_["demand"] = nlohmann::json::array({demand});
    return *this;
  }

  build_test_single_job skills(const std::vector<std::string>& skills) {
    content_["skills"] = nlohmann::json(skills);
    return *this;
  }

  nlohmann::json content() const { return content_; }

private:
  std::string type_;
  nlohmann::json content_ = {
    {"id", "job1"},
    {"places",
     {
       {
         type_,
         {
           {"location", nlohmann::json::array({1.0, 0.0})},
           {"duration", 1},
           {"times", nlohmann::json::array({nlohmann::json::array({"1970-01-01T00:00:00Z", "1970-01-01T00:16:40Z"})})},
         },
       },
     }},
    {
      "demand",
      {1},
    }};
};

/// Provides the way to create serialized representation of pickup job.
struct build_test_pickup_job final : public build_test_single_job {
  build_test_pickup_job() : build_test_single_job("pickup") {}
};

/// Provides the way to create serialized representation of delivery job.
struct build_test_delivery_job : public build_test_single_job {
  build_test_delivery_job() : build_test_single_job("delivery") {}
};

/// Provides the way to create serialized representation of shipment job.
struct build_test_shipment_job final {
  build_test_shipment_job id(nlohmann::json value) {
    content_["id"] = std::move(value);
    return *this;
  }

  build_test_shipment_job demand(int demand) {
    content_["demand"] = nlohmann::json::array({demand});
    return *this;
  }

  build_test_shipment_job pickup(nlohmann::json value) {
    content_["places"]["pickup"] = std::move(value);
    return *this;
  }

  build_test_shipment_job delivery(nlohmann::json value) {
    content_["places"]["delivery"] = std::move(value);
    return *this;
  }

  nlohmann::json content() { return content_; }

private:
  nlohmann::json content_ = {
    {"id", "job1"},
    {"places",
     {
       {
         "pickup",
         {
           {"location", nlohmann::json::array({1.0, 0.0})},
           {"duration", 1},
           {"times", nlohmann::json::array({nlohmann::json::array({"1970-01-01T00:00:00Z", "1970-01-01T00:16:40Z"})})},
         },
       },
       {
         "delivery",
         {
           {"location", nlohmann::json::array({2.0, 0.0})},
           {"duration", 1},
           {"times", nlohmann::json::array({nlohmann::json::array({"1970-01-01T00:00:00Z", "1970-01-01T00:16:40Z"})})},
         },
       },
     }},
    {
      "demand",
      nlohmann::json::array({1}),
    }};
};

/// Provides the way to build relation.
struct build_test_relation final {
  build_test_relation type(const std::string& value) {
    content_["type"] = value;
    return *this;
  }

  build_test_relation vehicle(const std::string& value) {
    content_["vehicleId"] = value;
    return *this;
  }

  build_test_relation jobs(std::initializer_list<std::string> value) {
    ranges::copy(value, ranges::back_inserter(content_["jobs"]));
    return *this;
  }

  nlohmann::json content() { return content_; }

private:
  nlohmann::json content_ = {{"type", "sequence"}, {"vehicleId", "vehicle_1"}, {"jobs", nlohmann::json::array()}};
};

/// Provides the way to create serialized representation of plan.
struct build_test_plan final {
  build_test_plan addJob(nlohmann::json job) {
    jobs_.push_back(std::move(job));
    return *this;
  }

  build_test_plan addRelation(nlohmann::json relation) {
    relations_.push_back(std::move(relation));
    return *this;
  }

  nlohmann::json content() const {
    nlohmann::json plan;

    plan["jobs"] = jobs_;
    if (!relations_.empty()) plan["relations"] = relations_;

    return plan;
  }

private:
  nlohmann::json jobs_ = nlohmann::json::array();
  nlohmann::json relations_ = nlohmann::json::array();
};

// endregion

// region Fleet
/// Provides the way to create serialized representation of vehicle type.
struct build_test_vehicle final {
  build_test_vehicle id(nlohmann::json value) {
    content_["id"] = std::move(value);
    return *this;
  }

  build_test_vehicle profile(nlohmann::json value) {
    content_["profile"] = std::move(value);
    return *this;
  }

  build_test_vehicle start(nlohmann::json value) {
    content_["places"]["start"] = std::move(value);
    return *this;
  }

  build_test_vehicle locations(std::pair<double, double> start, std::pair<double, double> end) {
    content_["places"]["start"]["location"] = nlohmann::json::array({start.first, start.second});
    content_["places"]["end"]["location"] = nlohmann::json::array({end.first, end.second});
    return *this;
  }

  build_test_vehicle places(nlohmann::json value) {
    content_["places"] = std::move(value);
    return *this;
  }

  build_test_vehicle end(nlohmann::json value) {
    content_["places"]["end"] = std::move(value);
    return *this;
  }

  build_test_vehicle costs(nlohmann::json value) {
    content_["costs"] = std::move(value);
    return *this;
  }

  build_test_vehicle capacity(int value) {
    content_["capacity"] = nlohmann::json::array({value});
    return *this;
  }

  build_test_vehicle amount(int value) {
    content_["amount"] = value;
    return *this;
  }

  build_test_vehicle setBreak(nlohmann::json value) {
    content_["break"] = std::move(value);

    return *this;
  }

  build_test_vehicle skills(std::vector<std::string> skills) {
    content_["skills"] = nlohmann::json(skills);
    return *this;
  }

  nlohmann::json content() const { return content_; }

private:
  nlohmann::json content_ = {
    {"id", "vehicle"},
    {"profile", "car"},
    {"costs", {{"distance", 1.0}, {"time", 1.0}, {"fixed", 10.0}}},
    {
      "places",
      {
        {"start", {{"time", "1970-01-01T00:00:00Z"}, {"location", nlohmann::json::array({0.0, 0.0})}}},
        {"end", {{"time", "1970-01-01T00:16:40"}, {"location", nlohmann::json::array({0.0, 0.0})}}},
      },
    },
    {"capacity", nlohmann::json::array({2})},
    {"amount", 2}};
};

/// Provides the way to create serialized representation of fleet.
struct build_test_fleet final {
  build_test_fleet addVehicle(nlohmann::json vehicle) {
    vehicles_.push_back(vehicle);
    return *this;
  }

  nlohmann::json content() const {
    nlohmann::json fleet;
    fleet["types"] = vehicles_;
    return fleet;
  }

private:
  nlohmann::json vehicles_ = nlohmann::json::array();
};
// endregion

// region Problem

/// Provides the way to create serialized representation of problem.
struct build_test_problem final {
  build_test_problem() { id("problem"); }

  build_test_problem id(std::string id) {
    problem_["id"] = std::move(id);
    return *this;
  }

  build_test_problem plan(build_test_plan plan) {
    problem_["plan"] = plan.content();
    return *this;
  }

  build_test_problem fleet(build_test_fleet fleet) {
    problem_["fleet"] = fleet.content();
    return *this;
  }

  build_test_problem matrices(nlohmann::json value) {
    // TODO generate matrices from plan and fleet
    problem_["matrices"] = std::move(value);
    return *this;
  }

  std::stringstream build() {
    std::stringstream ss;
    ss << problem_.dump(2);
    // std::cout << problem_.dump();
    return ss;
  }

private:
  nlohmann::json problem_;
};

// endregion
}