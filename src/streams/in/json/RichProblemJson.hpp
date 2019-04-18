#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/Problem.hpp"
#include "models/costs/MatrixTransportCosts.hpp"
#include "streams/in/json/detail/RichProblemParser.hpp"

#include <istream>
#include <map>
#include <range/v3/utility/variant.hpp>

namespace vrp::streams::in {

/// Parses rich VRP from json.
/// TODO Not yet fully implemented.
struct read_rich_json_type {
  std::shared_ptr<models::Problem> operator()(std::istream& input) const {
    using namespace vrp::algorithms::construction;

    nlohmann::json j;
    input >> j;

    auto problem = j.get<detail::rich::Problem>();

    auto transport = transportCosts(problem);
    auto activity = std::make_shared<models::costs::ActivityCosts>();

    auto fleet = readFleet(problem);
    auto jobs = readJobs(problem, *transport, *fleet);

    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, transport, activity))
      .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>());

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
  std::shared_ptr<models::problem::Fleet> readFleet(const detail::rich::Problem& problem) const {
    using namespace vrp::algorithms::construction;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;

    assert(problem.fleet.drivers.size() == 1);

    auto fleet = std::make_shared<Fleet>();
    ranges::for_each(problem.fleet.vehicles, [&](const auto& vehicle) {
      ranges::for_each(ranges::view::closed_indices(1, vehicle.amount), [&](auto index) {
        assert(vehicle.capabilities.has_value());
        assert(vehicle.capabilities.value().capacity.size() == 1);

        fleet->add(
          Vehicle{getProfile(vehicle.profile),

                  Costs{vehicle.costs.fixed,
                        vehicle.costs.distance,
                        vehicle.costs.driving,
                        vehicle.costs.waiting,
                        vehicle.costs.serving},

                  Dimensions{{"id", vehicle.id + "_" + std::to_string(index)},
                             {VehicleActivitySize<int>::DimKeyCapacity, vehicle.capabilities.value().capacity.front()}},

                  ranges::accumulate(vehicle.details, std::vector<Vehicle::Detail>{}, [](auto& acc, const auto detail) {
                    acc.push_back(Vehicle::Detail{detail.start, detail.end, {detail.time.start, detail.time.end}});
                    return std::move(acc);
                  })});
      });
    });

    ranges::for_each(problem.fleet.drivers, [&](const auto& driver) {
      ranges::for_each(ranges::view::closed_indices(1, driver.amount), [&](auto index) {
        using namespace vrp::models::common;
        using namespace vrp::models::problem;

        // TODO implement driver costs
        fleet->add(Driver{Costs{0, 0, 0, 0, 0}, Dimensions{{"id", driver.id + "_" + std::to_string(index)}}});
      });
    });

    return fleet;
  }

  std::shared_ptr<models::problem::Jobs> readJobs(const detail::rich::Problem& problem,
                                                  const models::costs::TransportCosts& transport,
                                                  const models::problem::Fleet& fleet) const {
    using namespace ranges;
    using namespace algorithms::construction;
    using namespace models::common;
    using namespace models::problem;

    auto jobs = view::for_each(problem.plan.jobs, [](const auto& job) {
      static auto ensureDemand = [](const auto& demand) {
        auto newDemand = demand.value_or(detail::rich::Demand{{{0}}, {{0}}});

        newDemand.pickup = newDemand.pickup.value_or(std::vector<int>{0});
        newDemand.delivery = newDemand.delivery.value_or(std::vector<int>{0});

        if (newDemand.pickup.value().empty()) newDemand.pickup.value().push_back(0);
        if (newDemand.delivery.value().empty()) newDemand.delivery.value().push_back(0);

        assert(newDemand.pickup.value().size() == 1 && newDemand.delivery.value().size() == 1);

        return std::move(newDemand);
      };

      static auto createService = [](const auto& s, const std::string& id) {
        assert(s.requirements.has_value());
        auto fixed = ensureDemand(s.requirements.value().demands.fixed);
        auto dynamic = ensureDemand(s.requirements.value().demands.dynamic);

        return std::make_shared<Service>(Service{
          // details
          ranges::accumulate(s.details,
                             std::vector<Service::Detail>{},
                             [](auto& acc, const auto detail) {
                               auto times = ranges::accumulate(
                                 detail.times, std::vector<TimeWindow>{}, [](auto& acc, const auto& time) {
                                   acc.push_back({time.start, time.end});
                                   return std::move(acc);
                                 });

                               acc.push_back(Service::Detail{detail.location, detail.duration, std::move(times)});
                               return std::move(acc);
                             }),
          // demand
          Dimensions{
            {"id", id},
            {VehicleActivitySize<int>::DimKeyDemand,
             VehicleActivitySize<int>::Demand{{fixed.pickup.value().front(), dynamic.pickup.value().front()},
                                              {fixed.delivery.value().front(), dynamic.delivery.value().front()}}}}});
      };

      auto result = job.variant.visit(ranges::overload(  //
        [&](const detail::rich::Service& s) -> models::problem::Job {
          return Job{ranges::emplaced_index<0>, createService(s, job.id)};
        },
        [&](const detail::rich::Sequence& s) -> models::problem::Job {
          // TODO use build_sequence as it adds specific dimens
          return models::problem::Job{
            ranges::emplaced_index<1>,
            std::make_shared<Sequence>(  //
              Sequence{ranges::accumulate(view::zip(s.services, view::iota(1)),
                                          std::vector<std::shared_ptr<const Service>>{},
                                          [&](auto& acc, const auto& pair) {
                                            const auto& [srv, index] = pair;

                                            acc.push_back(createService(srv, job.id + "_" + std::to_string(index)));
                                            return std::move(acc);
                                          }),
                       Dimensions{{"id", job.id}}})};
        }));
      return yield(result.index() == 0 ? ranges::get<0>(result) : ranges::get<1>(result));
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