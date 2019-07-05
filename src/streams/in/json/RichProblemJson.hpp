#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/Problem.hpp"
#include "models/costs/MatrixTransportCosts.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "streams/in/json/detail/CommonProblemConstraints.hpp"
#include "streams/in/json/detail/CoordIndex.hpp"
#include "streams/in/json/detail/RichProblemParser.hpp"
#include "utils/Date.hpp"

#include <gsl/gsl>
#include <istream>
#include <map>
#include <range/v3/utility/variant.hpp>

namespace vrp::streams::in {

/// Keeps tracks of unassigned jobs codes mapping.
struct RichProblemUnassignedCodes final {
  constexpr static int Time = 1;
  constexpr static int Size = 2;
};

/// Parses rich VRP from json.
/// TODO Not yet fully implemented.
struct read_rich_json_type {
  std::shared_ptr<models::Problem> operator()(std::istream& input) const {
    using namespace vrp::algorithms::construction;

    nlohmann::json j;
    input >> j;

    auto problem = j.get<detail::rich::Problem>();

    auto coordIndex = createCoordIndex(problem);

    auto transport = transportCosts(problem);
    auto activity = std::make_shared<models::costs::ActivityCosts>();

    auto fleet = readFleet(problem, coordIndex);
    auto jobs = readJobs(problem, *transport, *fleet, coordIndex);

    using Codes = RichProblemUnassignedCodes;
    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, transport, activity, Codes::Time))
      .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>(Codes::Size));

    return std::make_shared<models::Problem>(
      models::Problem{fleet,
                      jobs,
                      std::make_shared<std::vector<models::Lock>>(),  // TODO read initial routes
                      constraint,
                      std::make_shared<algorithms::objectives::penalize_unassigned_jobs<10000>>(),
                      activity,
                      transport,
                      {}});
  }

private:
  /// Creates coordinate index to match routing data.
  CoordIndex createCoordIndex(const detail::rich::Problem& problem) const {
    auto index = CoordIndex{};

    // analyze jobs
    const auto addService = [&index](const auto& service) {
      ranges::for_each(service.details, [&](const auto& detail) {
        if (detail.location) { index.add(detail.location.value().latitude, detail.location.value().longitude); }
      });
    };

    ranges::for_each(problem.plan.jobs, [&](const auto& job) {
      job.variant.visit(ranges::overload(
        [&](const detail::rich::Service& service) { addService(service); },
        [&](const detail::rich::Sequence& sequence) { ranges::for_each(sequence.services, addService); }));
    });

    // analyze fleet
    ranges::for_each(problem.fleet.drivers, [&](const auto& driver) {
      ranges::for_each(driver.availability, [&](const auto& availability) {
        if (availability.breakTime && availability.breakTime.value().location) {
          const auto& breakLoc = availability.breakTime.value().location.value();
          index.add(breakLoc.latitude, breakLoc.longitude);
        }
      });
    });

    ranges::for_each(problem.fleet.vehicles, [&](const auto& vehicle) {
      ranges::for_each(vehicle.availability, [&](const auto& availability) {
        index.add(availability.location.start.latitude, availability.location.start.longitude);

        if (availability.location.end) {
          const auto& endLoc = availability.location.end.value();
          index.add(endLoc.latitude, endLoc.longitude);
        }
      });
    });

    return std::move(index);
  }

  std::shared_ptr<models::problem::Fleet> readFleet(const detail::rich::Problem& problem,
                                                    const CoordIndex& coordIndex) const {
    using namespace vrp::algorithms::construction;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;

    Ensures(problem.fleet.drivers.size() == 1);

    auto dateParser = vrp::utils::parse_date_from_rc3339{};

    auto timeParser = [&](const auto& time) {
      return time ? TimeWindow{static_cast<double>(dateParser(time.value().start)),
                               static_cast<double>(dateParser(time.value().end))}
                  : TimeWindow{};
    };

    auto fleet = std::make_shared<Fleet>();
    ranges::for_each(problem.fleet.vehicles, [&](const auto& vehicle) {
      ranges::for_each(ranges::view::closed_indices(1, vehicle.amount), [&](auto index) {
        // TODO do not require capabilities to be present
        Expects(vehicle.capabilities.has_value());
        Expects(vehicle.capabilities.value().capacities.size() == 1);

        fleet->add(Vehicle{
          getProfile(vehicle.profile),

          Costs{vehicle.costs.fixed,
                vehicle.costs.distance,
                vehicle.costs.driving,
                vehicle.costs.waiting,
                vehicle.costs.serving},

          Dimensions{{"id", vehicle.id + "_" + std::to_string(index)},
                     {VehicleActivitySize<int>::DimKeyCapacity, vehicle.capabilities.value().capacities.front()}},

          ranges::accumulate(vehicle.availability, std::vector<Vehicle::Detail>{}, [&](auto& acc, const auto av) {
            acc.push_back(  //
              Vehicle::Detail{coordIndex.find(av.location.start.latitude, av.location.start.longitude),
                              av.location.end ? std::make_optional<Location>(Location{coordIndex.find(
                                                  av.location.end.value().latitude, av.location.end.value().longitude)})
                                              : std::make_optional<Location>(),
                              timeParser(av.time)});
            return std::move(acc);
          })});
      });
    });

    ranges::for_each(problem.fleet.drivers, [&](const auto& driver) {
      ranges::for_each(ranges::view::closed_indices(1, driver.amount), [&](auto index) {
        using namespace vrp::models::common;
        using namespace vrp::models::problem;

        // CONTINUE
        // TODO profiles, skills, vehicles

        fleet->add(
          Driver{Costs{driver.costs.fixed,
                       driver.costs.distance,
                       driver.costs.driving,
                       driver.costs.waiting,
                       driver.costs.serving},

                 Dimensions{{"id", driver.id + "_" + std::to_string(index)}},

                 ranges::accumulate(driver.availability, std::vector<Driver::Detail>{}, [&](auto& acc, const auto av) {
                   acc.push_back(Driver::Detail{timeParser(av.time)});
                   return std::move(acc);
                 })});
      });
    });

    return fleet;
  }

  std::shared_ptr<models::problem::Jobs> readJobs(const detail::rich::Problem& problem,
                                                  const models::costs::TransportCosts& transport,
                                                  const models::problem::Fleet& fleet,
                                                  const CoordIndex& coordIndex) const {
    using namespace ranges;
    using namespace algorithms::construction;
    using namespace models::common;
    using namespace models::problem;

    using SkillRawType = detail::common::SkillConstraint::RawType;

    auto dateParser = vrp::utils::parse_date_from_rc3339{};

    auto ensureDemand = [](const auto& demand) {
      auto newDemand = demand.value_or(detail::rich::Demand{{{0}}, {{0}}});

      newDemand.pickup = newDemand.pickup.value_or(std::vector<int>{0});
      newDemand.delivery = newDemand.delivery.value_or(std::vector<int>{0});

      if (newDemand.pickup.value().empty()) newDemand.pickup.value().push_back(0);
      if (newDemand.delivery.value().empty()) newDemand.delivery.value().push_back(0);

      Ensures(newDemand.pickup.value().size() == 1 && newDemand.delivery.value().size() == 1);

      return std::move(newDemand);
    };

    auto addSkillsIfPresent = [](const auto& requirements, Dimensions&& dimens) {
      if (requirements.has_value() && requirements.value().skills.has_value() &&
          !requirements.value().skills.value().empty()) {
        auto skills = std::make_shared<SkillRawType>(
          SkillRawType(requirements.value().skills.value().begin(), requirements.value().skills.value().end()));
        dimens.insert(std::make_pair("skills", skills));
      }

      return std::move(dimens);
    };

    auto createService = [&](const auto& s, const std::string& id) {
      Expects(s.requirements.has_value());
      auto fixed = ensureDemand(s.requirements.value().demands.fixed);
      auto dynamic = ensureDemand(s.requirements.value().demands.dynamic);

      return build_service{}
        .details(ranges::accumulate(
          s.details,
          std::vector<Service::Detail>{},
          [&](auto& out, const auto detail) {
            auto times = ranges::accumulate(detail.times, std::vector<TimeWindow>{}, [&](auto& in, const auto& time) {
              in.push_back(
                TimeWindow{static_cast<double>(dateParser(time.start)), static_cast<double>(dateParser(time.end))});
              return std::move(in);
            });

            out.push_back(Service::Detail{detail.location
                                            ? std::make_optional<Location>(Location{coordIndex.find(
                                                detail.location.value().latitude, detail.location.value().longitude)})
                                            : std::make_optional<Location>(),
                                          detail.duration,
                                          std::move(times)});
            return std::move(out);
          }))
        .dimens(addSkillsIfPresent(s.requirements,
                                   Dimensions{{"id", id},
                                              {VehicleActivitySize<int>::DimKeyDemand,
                                               VehicleActivitySize<int>::Demand{
                                                 {fixed.pickup.value().front(), dynamic.pickup.value().front()},
                                                 {fixed.delivery.value().front(), dynamic.delivery.value().front()}}}}))
        .shared();
    };

    auto jobs = view::for_each(problem.plan.jobs, [&](const auto& job) {
      if (job.variant.index() == 0)
        return yield(Job{ranges::emplaced_index<0>, createService(ranges::get<0>(job.variant), job.id)});

      return yield(models::problem::Job{
        ranges::emplaced_index<1>,
        build_sequence{}
          .dimens(Dimensions{{"id", job.id}})
          .services(ranges::accumulate(view::zip(ranges::get<1>(job.variant).services, view::iota(1)),
                                       std::vector<std::shared_ptr<Service>>{},
                                       [&](auto& acc, const auto& pair) {
                                         const auto& [srv, index] = pair;

                                         acc.push_back(createService(srv, job.id + "_" + std::to_string(index)));
                                         return std::move(acc);
                                       }))
          .shared()});
    });

    return std::make_shared<models::problem::Jobs>(models::problem::Jobs{transport, fleet, jobs});
  }

  std::shared_ptr<models::costs::MatrixTransportCosts> transportCosts(const detail::rich::Problem& problem) const {
    using namespace vrp::models::costs;

    auto durations = MatrixTransportCosts::DurationProfiles{};
    auto distances = MatrixTransportCosts::DistanceProfiles{};
    durations.resize(problem.routing.matrices.size());
    distances.resize(problem.routing.matrices.size());

    ranges::for_each(problem.routing.matrices, [&](const auto& matrix) {
      // TODO check that each profile is defined only once.
      auto profile = getProfile(matrix.profile);
      durations[profile] = std::move(matrix.durations);
      distances[profile] = std::move(matrix.distances);
    });

    return std::make_shared<MatrixTransportCosts>(MatrixTransportCosts{std::move(durations), std::move(distances)});
  }

  models::common::Profile getProfile(const std::string& value) const {
    if (value == "car") return 0;
    if (value == "truck") return 1;

    throw std::invalid_argument(std::string("Unknown routing profile: ") + value);
  }
};
}