# A "rich" json format

In order to support extra requirements like driver-vehicle match, multiday planning, etc., the following json format is proposed.

## Key points

* all features of here json format
* unlocks multiday planning
* adds driver entity to enable driver-vehicle matching algorithms as part of VRP solver
* supports jobs with variable locations (e.g. deliver in the morning to home or later to office)
* ...

### Fleet

Specifies information about available for planning resources.

#### Driver

To support driver-vehicle matching, a new _drivers_ property is introduced.

* _fleet.drivers.id_: unique driver type id
 
* _fleet.drivers.amount_: amount of drivers

* _fleet.drivers.availability_: array of properties defined as:
    * _location_ (optional): specifies start and end(optional) location of a driver.
    * _time_ (optional): specifies driver's shift time
    * _break_ (optional): specifies driver's break time with optional location

  One driver type can have more than one availability property which can be used for multiday planning.
    
* _fleet.drivers.costs_: driver costs per time and distance

* _fleet.drivers.capabilities_: array of properties defined as:
    * _skills_ (optional): a unique list of driver skills which might be required to perform a job
    * _profiles_ (optional): an array of vehicle profiles which driver is allowed to use
    * _vehicles_ (optional): an array of vehicle type ids which driver is allowed to use

#### Vehicle

* _fleet.vehicles.id_: unique vehicle type id
 
* _fleet.vehicless.amount_: amount of vehicles

* _fleet.vehicles.profile_: routing profile

* _fleet.vehicles.availability_: array of properties defined as:
    * _location_ (required): specifies start and end(optional) location of a vehicle
    * _time_ (optional): specifies vehicle's operational time
    
* _fleet.vehicles.costs_: vehicle costs per time and distance

* _fleet.vehicles.capabilities_: array of properties defined as:
    * _capacity_ (required): a vehicle cargo capacity
    * _facilities_ (optional): an array of available facilities
    
### Plan

Specifies information about what and how things should be performed.

#### Jobs

In general, there are two types of jobs:

* _service_: a single job which can model pickup, delivery, or service.
* _sequence_: a list of service jobs where all of them should be performed in non-strict sequential order or none.
    Can be used to model pickup/delivery or even pickup/delivery/delivery, etc. However, tend to be more computational expensive.

##### Service

All jobs with _type=service_ property.

* _plan.jobs.id_: service id
* _plan.jobs.details_: an array list of service details defined as:
   * _location_ (location): specifies job's location
   * _duration_ (required): specifies job's operation time 
   * _times_ (required): an array of time windows
* _plan.jobs.requirements_: specifies job requirements
    * _demands_: a various type of demands (fixed - picked from start or delivered to end, dynamic - picked and delivered along the route). 
    * _skills_: a driver skills requirements
    * _facilities_: a vehicle facilities requirements

##### Sequence

All jobs with _type=sequence_ property.

* _plan.jobs.id_: sequence id
* _plan.jobs.services_: an array of service jobs (without id and type properties)


### Routes

Routes object is used to put some additional constraints how jobs are assigned in relation to drivers and vehicles.

### TODO

* add necessary limits to driver and vehicle
* describe _routes_ objects
   
## Example

```
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
                "start": {"lat": 52, "lon": 13 },
                "end": {"lat": 52, "lon": 13 }
            },
            "time": { "start": "1970-01-01T00:00:00Z", "end": "1970-01-01T00:01:40Z" },
            "break": { 
                "time": {"start": "1970-01-01T00:00:40Z", "end": "1970-01-01T00:00:50Z"}, 
                "duration": 50, 
                "location": { "lat": 52.001, "lon": 13.001 }
            }
          }
        ],
        "costs": { "fixed": 0, "distance": 1, "driving": 1, "waiting": 1, "serving": 1 },
        "capabilities": {
          "skills": [],
          "profiles": ["car"],
          "vehicles": ["vehicle1"]
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
                "start": {"lat": 52, "lon": 13 },
                "end": {"lat": 52, "lon": 13 }
            },
            "time": { "start": "1970-01-01T00:00:00Z", "end": "1970-01-01T00:01:40Z" }
          }
        ],
        "costs": { "fixed": 0, "distance": 1, "driving": 1, "waiting": 1, "serving": 1 },
        "capabilities": {
          "capacity": [10],
          "facilities": []
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
            "location": {"lat": 52.1, "lon": 13.1 },
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
                "location": {"lat": 52.2, "lon": 13.2 },
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
                "location": {"lat": 52.3, "lon": 13.3 },
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
    ],
    "routes": [
      {
        "vehicleId": "vehicle1_1",
        "order": "sequence",
        "jobs": [
          "service1"
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

```