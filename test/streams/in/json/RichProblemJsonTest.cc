#include "streams/in/json/RichProblemJson.hpp"

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
        "amount": 1,
        "availability": [
          {
            "location": {
                "start": {"lat": 52.4862, "lon": 13.45148 },
                "end": {"lat": 52.4862, "lon": 13.45148 }
            },
            "time": { "start": "1970-01-01T00:00:00Z", "end": "1970-01-01T00:01:40Z" },
            "break": {
                "time": {"start": "1970-01-01T00:00:40Z", "end": "1970-01-01T00:00:50Z"},
                "duration": 50,
                "location": { "lat": 52.48315, "lon": 13.4330 }
            }
          }
        ],
        "costs": { "fixed": 0, "distance": 1, "driving": 1, "waiting": 1, "serving": 1 },
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
        "id": "vehicle1",
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
        "costs": { "fixed": 0, "distance": 1, "driving": 1, "waiting": 1, "serving": 1 },
        "capabilities": {
          "capacity": [10],
          "facilities": []
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
        "id": "service1",
        "type": "service",
        "details": [
          {
            "location": {"lat": 52.48325, "lon": 13.4436 },
            "duration": 0,
            "times": [
              {
                "start": "1970-01-01T00:00:00Z",
                "end": "1970-01-01T00:01:40Z"
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
        "id": "sequence1",
        "type": "sequence",
        "services": [
          {
            "details": [
              {
                "location": {"lat": 52.48300, "lon": 13.442 },
                "duration": 0,
                "times": [
                  {
                    "start": "1970-01-01T00:00:00Z",
                    "end": "1970-01-01T00:01:40Z"
                  }
                ]
              }
            ],
            "requirements": {
              "demands": {
                "dynamic": {
                  "pickup": [
                    1
                  ]
                }
              },
              "skills": []
            }
          },
          {
            "details": [
              {
                "location": {"lat": 52.48325, "lon": 13.4436 },
                "duration": 0,
                "times": [
                  {
                    "start": "1970-01-01T00:00:00Z",
                    "end": "1970-01-01T00:01:40Z"
                  }
                ]
              }
            ],
            "requirements": {
              "demands": {
                "dynamic": {
                  "delivery": [
                    1
                  ]
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
      THEN("creates problem") {
        REQUIRE(ranges::distance(problem->fleet->drivers()) == 1);
        REQUIRE(ranges::distance(problem->fleet->vehicles()) == 2);
        REQUIRE(ranges::distance(problem->fleet->profiles()) == 1);
        REQUIRE(problem->jobs->size() == 2);
      }
    }
  }
}
}
