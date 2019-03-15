#include "streams/in/json/HereJson.hpp"

#include <catch/catch.hpp>

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
                "demand": [1],
                "places": {
                    "pickup": {
                        "duration": 100,
                        "location": [52.48300, 13.4420]
                    },
                    "delivery": {
                        "duration": 100,
                        "location": [52.48325, 13.4436]
                    }
                }
            },
            {
                "id": "pickup_job",
                "demand": [1],
                "skills": ["unique2"],
                "places": {
                    "pickup": {
                        "duration": 100,
                        "location": [52.48321, 13.4438],
                        "times" : [["1970-01-01T00:00:00Z", "1970-01-01T00:01:40Z"]]
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
                    "time": 1
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
      THEN("creates problem") {
        REQUIRE(ranges::distance(problem->fleet->drivers()) == 1);
        REQUIRE(ranges::distance(problem->fleet->vehicles()) == 2);
        REQUIRE(ranges::distance(problem->fleet->profiles()) == 1);
        REQUIRE(problem->jobs->size() == 3);
      }
    }
  }
}
}
