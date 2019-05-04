#include "streams/in/json/HereProblemJson.hpp"

#include "algorithms/construction/InsertionSolutionContext.hpp"
#include "test_utils/models/Factories.hpp"

#include <any>
#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::models::common;

using Demand = VehicleActivitySize<int>::Demand;
using Skills = vrp::streams::in::detail::here::SkillConstraint::WrappedType;

namespace {

Job
getJobAt(size_t index, const Jobs& jobs) {
  auto v = jobs.all() | ranges::to_vector;
  return v.at(index);
}

std::shared_ptr<const Vehicle>
getVehicleAt(size_t index, const Fleet& fleet) {
  auto v = fleet.vehicles() | ranges::to_vector;
  return v.at(index);
}
}

namespace vrp::test {

SCENARIO("here json can read problem from stream", "[streams][in][json]") {
  GIVEN("simple json data") {
    std::stringstream ss;
    ss << R"(
{
    "id": "problemId",
    "plan": {
        "jobs": [
            {
                "id": "delivery_job",
                "demand": [1],
                "skills": ["unique1"],
                "places": {
                    "delivery": {
                        "duration": 100,
                        "location": [52.48325, 13.4436],
                        "times" : [
                          ["1970-01-01T00:00:00Z", "1970-01-01T00:01:40Z"],
                          ["1970-01-01T00:01:50Z", "1970-01-01T00:02:00Z"]
                        ]
                    }
                }
            },
            {
                "id": "shipment_job",
                "demand": [2],
                "places": {
                    "pickup": {
                        "duration": 110,
                        "location": [52.48300, 13.4420],
                        "times" : [["1970-01-01T00:00:10Z", "1970-01-01T00:00:30Z"]]
                    },
                    "delivery": {
                        "duration": 120,
                        "location": [52.48325, 13.4436],
                        "times" : [["1970-01-01T00:00:50Z", "1970-01-01T00:01:00Z"]]
                    }
                }
            },
            {
                "id": "pickup_job",
                "demand": [3],
                "skills": ["unique2"],
                "places": {
                    "pickup": {
                        "duration": 90,
                        "location": [52.48321, 13.4438],
                        "times" : [["1970-01-01T00:00:10Z", "1970-01-01T00:01:10Z"]]
                    }
                }
            }
        ],
      "relations": [
        {
          "type" : "tour",
          "vehicleId": "myVehicle_1",
          "jobs": ["pickup_job"]
        }
      ]
    },
    "fleet": {
        "types": [
            {
                "id": "myVehicle",
                "profile": "car",
                "amount": 2,
                "capacity": [10],
                "skills": ["unique1", "unique2"],
                "costs": {
                    "fixed": 100,
                    "distance": 1,
                    "time": 2
                },
                "places": {
                    "start": {
                        "location": [52.4862, 13.45148],
                        "time": "1970-01-01T00:00:00Z"
                    },
                    "end": {
                        "location": [52.4862, 13.45148],
                        "time": "1970-01-01T00:01:40Z"
                    }
                },
                "limits": {
                  "maxDistance": 123.1,
                  "shiftTime": 100
                },
                "break": {
                  "duration": 100,
                  "location": [52.48315, 13.4330],
                  "times" : [
                    ["1970-01-01T00:00:10Z", "1970-01-01T00:01:20Z"],
                    ["1970-01-01T00:01:00Z", "1970-01-01T00:03:00Z"]
                  ]
                }
            }
        ]
    },
  "matrices": [
    {
      "profile": "car",
      "durations": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      "distances": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    }
  ]
}
)";

    WHEN("read from stream") {
      auto problem = streams::in::read_here_json_type{}(ss);

      THEN("creates problem with expected plan size") { REQUIRE(problem->jobs->size() == 3 + 2); }

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
        REQUIRE(std::any_cast<std::string>(delivery->dimens.at("id")) == "delivery_job");
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).pickup.first == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).pickup.second == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).delivery.first == 1);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).delivery.second == 0);
        REQUIRE(std::any_cast<Skills>(delivery->dimens.at("skills"))->size() == 1);
      }

      THEN("creates expected shipment job") {
        auto shipment = ranges::get<1>(getJobAt(1, *problem->jobs));

        REQUIRE(shipment->services.size() == 2);
        REQUIRE(std::any_cast<std::string>(shipment->dimens.at("id")) == "shipment_job");

        auto pickup = shipment->services.at(0);
        REQUIRE(pickup->details.size() == 1);
        REQUIRE(pickup->details.front().location.value() == 1);
        REQUIRE(pickup->details.front().duration == 110);
        REQUIRE(pickup->details.front().times.size() == 1);
        REQUIRE(pickup->details.front().times.at(0).start == 10);
        REQUIRE(pickup->details.front().times.at(0).end == 30);
        REQUIRE(std::any_cast<std::string>(pickup->dimens.at("id")) == "shipment_job");
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).pickup.first == 0);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).pickup.second == 2);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).delivery.first == 0);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).delivery.second == 0);

        auto delivery = shipment->services.at(1);
        REQUIRE(delivery->details.size() == 1);
        REQUIRE(delivery->details.front().location.value() == 0);
        REQUIRE(delivery->details.front().duration == 120);
        REQUIRE(delivery->details.front().times.size() == 1);
        REQUIRE(delivery->details.front().times.at(0).start == 50);
        REQUIRE(delivery->details.front().times.at(0).end == 60);
        REQUIRE(std::any_cast<std::string>(delivery->dimens.at("id")) == "shipment_job");
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).pickup.first == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).pickup.second == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).delivery.first == 0);
        REQUIRE(std::any_cast<Demand>(delivery->dimens.at("demand")).delivery.second == 2);
      }

      THEN("creates expected pickup job") {
        auto pickup = ranges::get<0>(getJobAt(2, *problem->jobs));

        REQUIRE(pickup->details.size() == 1);
        REQUIRE(pickup->details.front().location.value() == 2);
        REQUIRE(pickup->details.front().duration == 90);
        REQUIRE(pickup->details.front().times.size() == 1);
        REQUIRE(pickup->details.front().times.at(0).start == 10);
        REQUIRE(pickup->details.front().times.at(0).end == 70);
        REQUIRE(std::any_cast<std::string>(pickup->dimens.at("id")) == "pickup_job");
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).pickup.first == 3);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).pickup.second == 0);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).delivery.first == 0);
        REQUIRE(std::any_cast<Demand>(pickup->dimens.at("demand")).delivery.second == 0);
      }

      THEN("creates problem with expected fleet size") {
        REQUIRE(ranges::distance(problem->fleet->drivers()) == 1);
        REQUIRE(ranges::distance(problem->fleet->vehicles()) == 2);
        REQUIRE(ranges::distance(problem->fleet->profiles()) == 1);
      }

      THEN("creates expected vehicles") {
        ranges::for_each(ranges::view::closed_indices(0, 1), [&](auto index) {
          auto vehicle = getVehicleAt(index, *problem->fleet);

          REQUIRE(std::any_cast<std::string>(vehicle->dimens.at("id")) ==
                  (std::string("myVehicle_") + std::to_string(index + 1)));
          REQUIRE(std::any_cast<Skills>(vehicle->dimens.at("skills"))->size() == 2);
          REQUIRE(vehicle->profile == 0);
          REQUIRE(vehicle->costs.fixed == 100);
          REQUIRE(vehicle->costs.perDistance == 1);
          REQUIRE(vehicle->costs.perDrivingTime == 2);
          REQUIRE(vehicle->costs.perWaitingTime == 2);
          REQUIRE(vehicle->costs.perServiceTime == 2);
          REQUIRE(vehicle->details.size() == 1);
          REQUIRE(vehicle->details.front().start == 3);
          REQUIRE(vehicle->details.front().end.value() == 3);
          REQUIRE(vehicle->details.front().time.start == 0);
          REQUIRE(vehicle->details.front().time.end == 100);
        });
      }

      THEN("once accepted with no routes, breaks should be moved to ignored") {
        auto fleet = Fleet{};
        fleet  //
          .add(test_build_driver{}.owned())
          .add(test_build_vehicle{}
                 .dimens({{"id", std::string("myVehicle_1")}, {"typeId", std::string("myVehicle")}})
                 .owned());
        auto registry = std::make_shared<Registry>(fleet);
        auto ctx = InsertionSolutionContext{problem->jobs->all(), {}, {}, {}, registry};
        problem->constraint->accept(ctx);

        REQUIRE(ctx.required.size() == 3);
        REQUIRE(ctx.ignored.size() == 2);
        REQUIRE(std::any_cast<std::string>(ranges::get<0>(ctx.ignored.at(0))->dimens.at("type")) == "break");
        REQUIRE(std::any_cast<std::string>(ranges::get<0>(ctx.ignored.at(1))->dimens.at("type")) == "break");
      }
    }
  }
}
}
