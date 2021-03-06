#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/ActorJobLock.hpp"
#include "algorithms/construction/constraints/ActorTravelLimit.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/Problem.hpp"
#include "models/costs/MatrixTransportCosts.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "streams/in/json/detail/CoordIndex.hpp"
#include "streams/in/json/detail/HereProblemConstraints.hpp"
#include "streams/in/json/detail/HereProblemParser.hpp"
#include "utils/Date.hpp"

#include <gsl/gsl>
#include <istream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <range/v3/utility/variant.hpp>
#include <tuple>
#include <unordered_map>

namespace vrp::streams::in {

/// Keeps tracks of unassigned jobs codes mapping.
struct HereProblemUnassignedCodes final {
  constexpr static int Time = 1;
  constexpr static int Size = 2;
  constexpr static int Lock = 3;
  constexpr static int Break = 4;

  constexpr static int DistanceLimit = 5;
  constexpr static int DurationLimit = 6;

  constexpr static int Skill = 10;
  constexpr static int Reachable = 11;
};

/// Parses HERE VRP problem definition from json.
struct read_here_json_type {
private:
  using JobIndex = std::unordered_map<std::string, models::problem::Job>;
  using Limit = algorithms::construction::ActorTravelLimit::Limit;

public:
  std::shared_ptr<models::Problem> operator()(std::istream& input) const {
    using namespace vrp::algorithms::construction;
    using namespace vrp::models;

    nlohmann::json j;
    input >> j;

    auto problem = j.get<detail::here::Problem>();

    auto coordIndex = createCoordIndex(problem);
    auto jobIndex = JobIndex{};

    auto transport = transportCosts(problem);
    auto activity = std::make_shared<costs::ActivityCosts>();

    auto fleet = readFleet(problem, coordIndex);
    auto jobs = readJobs(problem, coordIndex, *transport, *fleet, jobIndex);
    auto locks = readLocks(problem, *jobs, jobIndex);
    auto limits = readLimits(problem);

    using Codes = HereProblemUnassignedCodes;
    auto constraint = std::make_shared<InsertionConstraint>();
    if (!locks->empty()) constraint->addHard<ActorJobLock>(std::make_shared<ActorJobLock>(*locks, Codes::Lock));

    constraint->addHardActivity(std::make_shared<detail::here::ReachableConstraint>(transport, Codes::Reachable))
      .add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, transport, activity, Codes::Time))
      .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>(Codes::Size))
      .addHardActivity(std::make_shared<detail::here::BreakConstraint>(Codes::Break))
      .addHardRoute(std::make_shared<detail::here::SkillConstraint>(Codes::Skill));

    if (!limits.empty())
      constraint->addHardActivity(
        std::make_shared<ActorTravelLimit>(limits, transport, activity, Codes::DistanceLimit, Codes::DurationLimit));

    auto extras = std::make_shared<std::map<std::string, std::any>>();
    extras->insert(std::make_pair("coordIndex", std::move(coordIndex)));
    extras->insert(std::make_pair("problemId", problem.id));

    return std::make_shared<models::Problem>(
      models::Problem{fleet,
                      jobs,
                      locks,  // TODO read relations
                      constraint,
                      std::make_shared<algorithms::objectives::penalize_unassigned_jobs<10000>>(),
                      activity,
                      transport,
                      extras});
  }

private:
  /// Creates coordinate index to match routing data.
  CoordIndex createCoordIndex(const detail::here::Problem& problem) const {
    auto index = CoordIndex{};
    ranges::for_each(problem.plan.jobs, [&](const auto& variant) {
      detail::here::analyze_variant<void>(
        variant,
        [&index](const detail::here::Job& job) {
          if (job.places.pickup) index.add(job.places.pickup.value().location);
          if (job.places.delivery) index.add(job.places.delivery.value().location);
        },
        [&index](const detail::here::MultiJob& multi) {
          ranges::for_each(multi.places.pickups, [&](const auto& pickup) { index.add(pickup.location); });
          ranges::for_each(multi.places.deliveries, [&](const auto& delivery) { index.add(delivery.location); });
        });
    });

    ranges::for_each(problem.fleet.types, [&](const auto& vehicle) {
      index.add(vehicle.places.start.location);
      if (vehicle.places.end) index.add(vehicle.places.end.value().location);
      if (vehicle.vehicleBreak && vehicle.vehicleBreak.value().location)
        index.add(vehicle.vehicleBreak.value().location.value());
    });

    return std::move(index);
  }

  std::shared_ptr<models::problem::Fleet> readFleet(const detail::here::Problem& problem,
                                                    const CoordIndex& coordIndex) const {
    using namespace vrp::algorithms::construction;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;
    using namespace vrp::utils;

    using SkillWrappedType = detail::here::SkillConstraint::WrappedType;
    using SkillRawType = detail::here::SkillConstraint::RawType;

    auto dateParser = parse_date_from_rc3339{};
    auto fleet = std::make_shared<Fleet>();

    ranges::for_each(problem.fleet.types, [&](const auto& vehicle) {
      Expects(vehicle.capacity.size() == 1);

      auto details = std::vector<Vehicle::Detail>{
        Vehicle::Detail{coordIndex.find(vehicle.places.start.location),
                        vehicle.places.end ? std::make_optional(coordIndex.find(vehicle.places.end.value().location))
                                           : std::optional<models::common::Location>{},
                        TimeWindow{static_cast<double>(dateParser(vehicle.places.start.time)),
                                   vehicle.places.end ? static_cast<double>(dateParser(vehicle.places.end.value().time))
                                                      : std::numeric_limits<double>::max()}}};

      SkillWrappedType skills = {};
      if (vehicle.skills.has_value() && !vehicle.skills.value().empty()) {
        skills =
          std::make_shared<SkillRawType>(SkillRawType(vehicle.skills.value().begin(), vehicle.skills.value().end()));
      }

      ranges::for_each(ranges::view::closed_indices(1, vehicle.amount), [&](auto index) {
        auto dimens = Dimensions{{"typeId", vehicle.id},
                                 {"id", vehicle.id + "_" + std::to_string(index)},
                                 {VehicleActivitySize<int>::DimKeyCapacity, vehicle.capacity.front()}};

        if (skills) dimens.insert(std::make_pair("skills", skills));

        fleet->add(Vehicle{getProfile(vehicle.profile),
                           Costs{vehicle.costs.fixed.value_or(0),
                                 vehicle.costs.distance,
                                 vehicle.costs.time,
                                 vehicle.costs.time,
                                 vehicle.costs.time},
                           dimens,
                           details});
      });
    });

    fleet->add(Driver{Costs{0, 0, 0, 0, 0}, Dimensions{{"id", "driver"}}});

    return fleet;
  }

  std::shared_ptr<models::problem::Jobs> readJobs(const detail::here::Problem& problem,
                                                  const CoordIndex& coordIndex,
                                                  const models::costs::TransportCosts& transport,
                                                  const models::problem::Fleet& fleet,
                                                  JobIndex& jobIndex) const {
    return std::make_shared<models::problem::Jobs>(models::problem::Jobs{
      transport,
      fleet,
      ranges::view::concat(readRequiredJobs(problem, coordIndex, jobIndex),
                           readConditionalJobs(problem, coordIndex, jobIndex)),
    });
  }

  ranges::any_view<models::problem::Job> readRequiredJobs(const detail::here::Problem& problem,
                                                          const CoordIndex& coordIndex,
                                                          JobIndex& jobIndex) const {
    using namespace ranges;
    using namespace algorithms::construction;
    using namespace models::common;
    using namespace models::problem;

    using SkillRawType = detail::here::SkillConstraint::RawType;

    static auto getDemand = [](const auto& demand, bool isPickup, bool isFixed) {
      Expects(demand.size() == 1);
      auto value = demand.front();
      return VehicleActivitySize<int>::Demand{{isFixed && isPickup ? value : 0, !isFixed && isPickup ? value : 0},
                                              {isFixed && !isPickup ? value : 0, !isFixed && !isPickup ? value : 0}};
    };

    static auto addSkillsIfPresent = [](const auto& job, Dimensions&& dimens, bool skipSkills) {
      if (!skipSkills && job.skills.has_value() && !job.skills.value().empty()) {
        auto skills =
          std::make_shared<SkillRawType>(SkillRawType(job.skills.value().begin(), job.skills.value().end()));
        dimens.insert(std::make_pair("skills", skills));
      }

      return std::move(dimens);
    };

    auto dateParser = vrp::utils::parse_date_from_rc3339{};

    return view::for_each(problem.plan.jobs, [&, dateParser](const auto& variant) {
      auto withIndex = [&](const std::string& id, const auto& j) {
        jobIndex[id] = j;
        return j;
      };

      auto createService = [&, dateParser](
                             const auto& job, const auto& place, const auto& demand, bool skipSkills = false) {
        auto times = place.times.has_value()  //
          ? ranges::accumulate(place.times.value(),
                               std::vector<TimeWindow>{},
                               [&](auto& acc, const auto& time) {
                                 acc.push_back({static_cast<double>(dateParser(time.at(0))),
                                                static_cast<double>(dateParser(time.at(1)))});
                                 return std::move(acc);
                               })
          : std::vector<TimeWindow>{TimeWindow{0, std::numeric_limits<double>::max()}};

        auto dimens = Dimensions{{"id", job.id}, {VehicleActivitySize<int>::DimKeyDemand, demand}};
        if (place.tag.has_value()) dimens["tag"] = place.tag.value();

        return std::make_shared<Service>(
          Service{{Service::Detail{coordIndex.find(place.location), place.duration, std::move(times)}},
                  addSkillsIfPresent(job, std::move(dimens), skipSkills)});
      };

      return ranges::yield(detail::here::analyze_variant<models::problem::Job>(
        variant,
        [&](const detail::here::Job& job) {
          Expects(job.places.pickup || job.places.delivery);

          // shipment
          if (job.places.pickup && job.places.delivery) {
            return withIndex(
              job.id,
              as_job(
                build_sequence{}
                  .dimens(addSkillsIfPresent(job, Dimensions{{"id", job.id}}, false))
                  .service(createService(job, job.places.pickup.value(), getDemand(job.demand, true, false), true))
                  .service(createService(job, job.places.delivery.value(), getDemand(job.demand, false, false), true))
                  .shared()));
            // pickup
          } else if (job.places.pickup) {
            return withIndex(job.id,
                             as_job(createService(job, job.places.pickup.value(), getDemand(job.demand, true, true))));
          }

          // delivery
          return withIndex(job.id,
                           as_job(createService(job, job.places.delivery.value(), getDemand(job.demand, false, true))));
        },
        [&](const detail::here::MultiJob& job) {
          auto sequence = build_sequence{};

          ranges::for_each(job.places.pickups, [&](const auto& pickup) {
            sequence.service(createService(job, pickup, getDemand(pickup.demand, true, false), true));
          });
          ranges::for_each(job.places.deliveries, [&](const auto& delivery) {
            sequence.service(createService(job, delivery, getDemand(delivery.demand, false, false), true));
          });

          // TODO add permutation function to dimens

          return withIndex(
            job.id, as_job(sequence.dimens(addSkillsIfPresent(job, Dimensions{{"id", job.id}}, false)).shared()));
        }));
    });
  }

  ranges::any_view<models::problem::Job> readConditionalJobs(const detail::here::Problem& problem,
                                                             const CoordIndex& coordIndex,
                                                             JobIndex& jobIndex) const {
    using namespace ranges;
    using namespace algorithms::construction;
    using namespace models::common;
    using namespace models::problem;

    auto dateParser = vrp::utils::parse_date_from_rc3339{};

    // convert vehicle breaks to conditional jobs
    return view::for_each(
      problem.fleet.types | ranges::view::filter([](const auto& v) { return v.vehicleBreak.has_value(); }),
      [&, dateParser](const auto& vehicle) {
        const auto& model = vehicle.vehicleBreak.value();
        const auto id = vehicle.id;
        const auto detail = Service::Detail{
          model.location ? std::make_optional<Location>(Location{coordIndex.find(model.location.value())})
                         : std::make_optional<Location>(),
          model.duration,
          ranges::accumulate(model.times, std::vector<TimeWindow>{}, [&](auto& acc, const auto& time) {
            acc.push_back({static_cast<double>(dateParser(time.at(0))), static_cast<double>(dateParser(time.at(1)))});
            return std::move(acc);
          })};

        return view::for_each(ranges::view::closed_indices(1, vehicle.amount), [=, &jobIndex](auto index) {
          auto job = as_job(build_service{}
                              .details({detail})
                              .dimens(Dimensions{{"id", std::string("break")},
                                                 {"type", std::string("break")},
                                                 {"vehicleId", id + "_" + std::to_string(index)}})
                              .shared());
          jobIndex[id + "_" + std::to_string(index) + "_break"] = job;
          return yield(job);
        });
      });
  }

  std::shared_ptr<models::costs::MatrixTransportCosts> transportCosts(const detail::here::Problem& problem) const {
    using namespace vrp::models::costs;

    auto durations = MatrixTransportCosts::DurationProfiles{};
    auto distances = MatrixTransportCosts::DistanceProfiles{};
    durations.resize(problem.matrices.size());
    distances.resize(problem.matrices.size());

    ranges::for_each(problem.matrices, [&](const auto& matrix) {
      // TODO check that each profile is defined only once.
      auto profile = getProfile(matrix.profile);
      durations[profile] = std::move(matrix.durations);
      distances[profile] = std::move(matrix.distances);
    });

    return std::make_shared<MatrixTransportCosts>(MatrixTransportCosts{std::move(durations), std::move(distances)});
  }

  std::shared_ptr<std::vector<models::Lock>> readLocks(const detail::here::Problem& problem,
                                                       const models::problem::Jobs& jobs,
                                                       const JobIndex& jIndx) const {
    using namespace vrp::models;
    using namespace vrp::streams::in::detail::here;
    using namespace ranges;

    auto locks = std::make_shared<std::vector<Lock>>();

    if (!problem.plan.relations || problem.plan.relations.value().empty()) return locks;

    auto relations = ranges::accumulate(problem.plan.relations.value(),
                                        std::map<std::string, std::vector<detail::here::Relation>>{},
                                        [](auto& acc, const auto& relation) {
                                          Expects(!relation.jobs.empty());
                                          acc[relation.vehicleId].push_back(relation);
                                          return std::move(acc);
                                        });

    return ranges::accumulate(relations, locks, [&](auto& acc, const auto& pair) {
      auto vehicleId = pair.first;

      auto condition = [vehicleId = vehicleId](const auto& a) {
        return problem::get_vehicle_id{}(*a.vehicle) == vehicleId;
      };

      auto details = ranges::accumulate(pair.second, std::vector<Lock::Detail>{}, [&](auto& dtls, const auto& rel) {
        auto order = rel.type == RelationType::Tour
          ? Lock::Order::Any
          : (rel.type == RelationType::Flexible ? Lock::Order::Sequence : Lock::Order::Strict);

        auto position = Lock::Position{rel.jobs.front() == "departure", rel.jobs.back() == "arrival"};

        auto jobs = rel.jobs |  //
          view::filter([](const auto& j) { return j != "departure" && j != "arrival"; }) |
          view::transform([&](const auto& j) { return j == "break" ? jIndx.at(vehicleId + "_break") : jIndx.at(j); }) |
          ranges::to_vector;

        dtls.push_back(Lock::Detail{order, position, std::move(jobs)});

        return std::move(dtls);
      });

      acc->push_back(models::Lock{condition, std::move(details)});
      return acc;
    });
  }

  std::vector<Limit> readLimits(const detail::here::Problem& problem) const {
    using namespace ranges;

    return ranges::accumulate(  //
      problem.fleet.types | view::filter([](const auto& v) {
        return v.limits && (v.limits.value().maxDistance || v.limits.value().shiftTime);
      }),
      std::vector<Limit>{},
      [&](auto& acc, const auto& v) {
        auto typeId = v.id;
        acc.push_back(Limit{[typeId = typeId](const auto& a) {
                              return std::any_cast<std::string>(a.vehicle->dimens.find("typeId")->second) == typeId;
                            },
                            v.limits.value().maxDistance,
                            v.limits.value().shiftTime});
        return acc;
      });
  }

  models::common::Profile getProfile(const std::string& value) const {
    if (value == "car") return 0;
    if (value == "truck") return 1;

    throw std::invalid_argument(std::string("Unknown routing profile: ") + value);
  }
};
}