#pragma once

#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "streams/out/json/HereSolutionJson.hpp"
#include "test_utils/scenarios/here/Helpers.hpp"

#include <catch/catch.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <range/v3/all.hpp>

namespace vrp::test::here {

/// Checks whether passed solution is the same ignoring node order.
inline void
assertSolution(const std::shared_ptr<const models::Problem>& problem,
               const models::EstimatedSolution& estimatedSolution,
               const std::string& expected) {
  static auto sortToursByVehicleId = [](auto& json) {
    auto& tours = json["tours"];
    std::sort(tours.begin(), tours.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.at("vehicleId") < rhs.at("vehicleId");
    });
  };

  static auto sortUnassignedByJobId = [](auto& json) {
    auto& unassigned = json["unassigned"];
    std::sort(unassigned.begin(), unassigned.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.at("jobId") < rhs.at("jobId");
    });
  };

  auto resultJson = getSolutionAsJson(problem, estimatedSolution);
  auto expectedJson = nlohmann::json::parse(expected);

  sortToursByVehicleId(resultJson);
  sortToursByVehicleId(expectedJson);

  sortUnassignedByJobId(resultJson);
  sortUnassignedByJobId(expectedJson);

  std::cout << nlohmann::json::diff(resultJson, expectedJson).dump(2);

  REQUIRE(resultJson == expectedJson);
}

/// Checks whether shipment activities are correctly defined in tour.
inline void
assertShipment(const nlohmann::json& shipment, const nlohmann::json& tour) {
  using namespace ranges;

  auto shipmentActivities = getJobActivitiesFromTour(shipment, tour);

  REQUIRE(shipmentActivities.size() == 2);
  REQUIRE(shipmentActivities.front()["type"] == "pickup");
  REQUIRE(shipmentActivities.back()["type"] == "delivery");
}
}