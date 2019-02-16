#include "streams/in/RichJson.hpp"

#include <catch/catch.hpp>

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
        "id": "driver1",
        "amount" : 1
      }
    ],
    "vehicles": [
      {
        "id": "vehicle1",
        "profile" : "car",
        "details" : [
          {"start": 0, "end": 0, "time" : { "start": 0, "end" : 1000} }
        ],
        "costs": {
          "fixed" : 0,
          "distance" : 1,
          "driving" : 1,
          "waiting": 1,
          "serving": 1
        },
        "capabilities": {
          "capacity" : [10],
          "skills": []
        },
        "amount" : 2
      }
    ]
  },
  "plan": {
    "jobs" : [
      {
        "id": "service1",
        "type": "service",
        "details": [
          { "location" : 1, "duration" : 0, "times": [{ "start" :  0, "end" : 1000 }] }
        ],
        "requirements": {
          "demands": { "fixed": { "delivery": [1] } },
          "skills": []
        }
      },
      {
        "id": "sequence1",
        "type": "sequence",
        "services": [
          {
            "details": [
              { "location" : 2, "duration" : 0, "times": [{ "start" :  0, "end" : 1000 }] }
            ],
            "requirements": {
              "demands": { "dynamic": { "pickup" :[1] } },
              "skills": []
            }
          },
          {
            "details": [
              { "location" : 3, "duration" : 0, "times": [{ "start" :  0, "end" : 1000 }] }
            ],
            "requirements": {
              "demands": { "dynamic": { "delivery": [1] } },
              "skills": []
            }
          }
        ]
      }
    ],
    "routes": [
      {
        "vehicleId": "vehicle1_1",
        "type": "sequence",
        "jobs": ["service1"]
      }
    ]
  },
  "routing": {
    "matrices": [
      {
        "profile": "car",
        "distances": [0,1,0,1],
        "durations": [0,1,0,1]
      }
    ]
  }
}
)";

    WHEN("read from stream") {
      auto problem = streams::in::read_rich_json_type{}(ss);
      THEN("creates problem") {
        // TODO
      }
    }
  }
}
}
