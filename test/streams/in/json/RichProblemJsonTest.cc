#include "streams/in/json/RichProblemJson.hpp"

#include "test_utils/streams/ProblemAnalyzer.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::problem;

using Demand = VehicleActivitySize<int>::Demand;

namespace vrp::test {

SCENARIO("rich json can read problem from stream", "[streams][in][json]") {
  GIVEN("simple json data") {
    std::stringstream ss;
    ss << R"(
{
  "id": "problemId",
  "fleet": {
    "drivers": [
      {
        "id": "myDriver",
        "amount": 1,
        "availability": [
          {
            "time": { "start": "1970-01-01T00:00:00Z", "end": "1970-01-01T00:01:40Z" },
            "break": {
                "time": {"start": "1970-01-01T00:00:40Z", "end": "1970-01-01T00:00:50Z"},
                "duration": 50,
                "location": { "lat": 52.48315, "lon": 13.4330 }
            }
          }
        ],
        "costs": { "fixed": 101, "distance": 10, "driving": 20, "waiting": 20, "serving": 20 },
        "capabilities": {
          "skills": [],
          "profiles": ["car"],
          "vehicles": ["vehicle1"]
        },
        "limits": {
          "maxTime": 100
        }
      }
    ],
    "vehicles": [
      {
        "id": "myVehicle",
        "amount": 2,
        "profile": "car",
        "availability": [
          {
            "location": {
                "start": {"lat": 52.4862, "lon": 13.45148 },
                "end": {"lat": 52.4862, "lon": 13.45148 }
            },
            "time": { "start": "1970-01-01T00:00:00Z", "end": "1970-01-01T00:01:40Z" }
          }
        ],
        "costs": { "fixed": 100, "distance": 1, "driving": 2, "waiting": 2, "serving": 2 },
        "capabilities": {
          "capacities": [10],
          "facilities": ["fridge"]
        },
        "limits": {
          "maxDistance": 100
        }
      }
    ]
  },
  "plan": {
    "jobs": [
      {
        "id": "service_delivery_job",
        "type": "service",
        "details": [
          {
            "location": {"lat": 52.4725, "lon": 13.456 },
            "duration": 100,
            "times": [
              {
                "start": "1970-01-01T00:00:00Z",
                "end": "1970-01-01T00:01:40Z"
              },
              {
                "start": "1970-01-01T00:01:50Z",
                "end": "1970-01-01T00:02:00Z"
              }
            ]
          }
        ],
        "requirements": {
          "demands": {
            "fixed": { "delivery": [1] }
          },
          "skills": [],
          "facilities": []
        }
      },
      {
        "id": "sequence_shipment_job",
        "type": "sequence",
        "services": [
          {
            "details": [
              {
                "location": {"lat": 52.48300, "lon": 13.442 },
                "duration": 110,
                "times": [
                  {
                    "start": "1970-01-01T00:00:10Z",
                    "end": "1970-01-01T00:00:30Z"
                  }
                ]
              }
            ],
            "requirements": {
              "demands": {
                "dynamic": {
                  "pickup": [2]
                }
              },
              "skills": []
            }
          },
          {
            "details": [
              {
                "location": {"lat": 52.4925, "lon": 13.4436 },
                "duration": 120,
                "times": [
                  {
                    "start": "1970-01-01T00:00:50Z",
                    "end": "1970-01-01T00:01:00Z"
                  }
                ]
              }
            ],
            "requirements": {
              "demands": {
                "dynamic": {
                  "delivery": [2]
                }
              },
              "skills": []
            }
          }
        ]
      }
    ]
  },
  "routing": {
    "matrices": [
      {
        "profile": "car",
        "distances": [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        "durations": [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
      }
    ]
  }
}
)";

    WHEN("read from stream") {
      auto problem = streams::in::read_rich_json_type{}(ss);
      THEN("creates expected problem size") {
        REQUIRE(ranges::distance(problem->fleet->drivers()) == 1);
        REQUIRE(ranges::distance(problem->fleet->vehicles()) == 2);
        REQUIRE(ranges::distance(problem->fleet->profiles()) == 1);
        REQUIRE(problem->jobs->size() == 2);
      }

      THEN("creates expected delivery job") {
        auto delivery = ranges::get<0>(getJobAt(0, *problem->jobs));

        REQUIRE(delivery->details.size() == 1);
        REQUIRE(delivery->details.front().location.value() == 0);
        REQUIRE(delivery->details.front().duration == 100);
        REQUIRE(delivery->details.front().times.size() == 2);
        REQUIRE(delivery->details.front().times.at(0).start == 0);
        REQUIRE(delivery->details.front().times.at(0).end == 100);
        REQUIRE(delivery->details.front().times.at(1).start == 110);
        REQUIRE(delivery->details.front().times.at(1).end == 120);
        REQUIRE(std::any_cast<std::string>(delivery->dimens.at("id")) == "service_delivery_job");
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).pickup.first == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).pickup.second == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).delivery.first == 1);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).delivery.second == 0);
        // TODO
        // REQUIRE(std::any_cast<Skills>(delivery->dimens.at("skills"))->size() == 1);
      }

      THEN("creates expected shipment job") {
        auto shipment = ranges::get<1>(getJobAt(1, *problem->jobs));

        REQUIRE(shipment->services.size() == 2);
        REQUIRE(std::any_cast<std::string>(shipment->dimens.at("id")) == "sequence_shipment_job");

        auto pickup = shipment->services.at(0);
        REQUIRE(pickup->details.size() == 1);
        REQUIRE(pickup->details.front().location.value() == 1);
        REQUIRE(pickup->details.front().duration == 110);
        REQUIRE(pickup->details.front().times.size() == 1);
        REQUIRE(pickup->details.front().times.at(0).start == 10);
        REQUIRE(pickup->details.front().times.at(0).end == 30);
        REQUIRE(std::any_cast<std::string>(pickup->dimens.at("id")) == "sequence_shipment_job_1");
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).pickup.first == 0);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).pickup.second == 2);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).delivery.first == 0);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).delivery.second == 0);

        auto delivery = shipment->services.at(1);
        REQUIRE(delivery->details.size() == 1);
        REQUIRE(delivery->details.front().location.value() == 2);
        REQUIRE(delivery->details.front().duration == 120);
        REQUIRE(delivery->details.front().times.size() == 1);
        REQUIRE(delivery->details.front().times.at(0).start == 50);
        REQUIRE(delivery->details.front().times.at(0).end == 60);
        REQUIRE(std::any_cast<std::string>(delivery->dimens.at("id")) == "sequence_shipment_job_2");
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).pickup.first == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).pickup.second == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).delivery.first == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).delivery.second == 2);
      }

      THEN("creates expected vehicles") {
        ranges::for_each(ranges::view::closed_indices(0, 1), [&](auto index) {
          auto vehicle = getVehicleAt(index, *problem->fleet);

          REQUIRE(std::any_cast<std::string>(vehicle->dimens.at("id")) ==
                  (std::string("myVehicle_") + std::to_string(index + 1)));
          // TODO
          // REQUIRE(std::any_cast<Skills>(vehicle->dimens.at("skills"))->size() == 2);
          REQUIRE(vehicle->profile == 0);
          REQUIRE(vehicle->costs.fixed == 100);
          REQUIRE(vehicle->costs.perDistance == 1);
          REQUIRE(vehicle->costs.perDrivingTime == 2);
          REQUIRE(vehicle->costs.perWaitingTime == 2);
          REQUIRE(vehicle->costs.perServiceTime == 2);
          REQUIRE(vehicle->details.size() == 1);
          REQUIRE(vehicle->details.front().time.start == 0);
          REQUIRE(vehicle->details.front().time.end == 100);
        });
      }
    }
  }
}
}
