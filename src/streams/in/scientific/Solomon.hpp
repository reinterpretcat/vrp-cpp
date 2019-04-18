#pragma once

#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "streams/in/extensions/SolomonCosts.hpp"

#include <algorithm>
#include <functional>
#include <istream>
#include <range/v3/all.hpp>
#include <sstream>
#include <tuple>

namespace vrp::streams::in {

/// Reads problem represented by classical solomon definition from stream.
template<typename Distance = cartesian_distance>
struct read_solomon_type final {
  constexpr static auto IdDimKey = "id";
  inline static auto DimKeyCapacity = algorithms::construction::VehicleActivitySize<int>::DimKeyCapacity;
  inline static auto DimKeyDemand = algorithms::construction::VehicleActivitySize<int>::DimKeyDemand;

  using Capacity = algorithms::construction::VehicleActivitySize<int>::Capacity;
  using Demand = algorithms::construction::VehicleActivitySize<int>::Demand;

  std::shared_ptr<models::Problem> operator()(std::istream& input) const {
    using namespace algorithms::construction;

    auto matrix = std::make_shared<RoutingMatrix<Distance>>();
    auto fleet = std::make_shared<models::problem::Fleet>();
    auto activity = std::make_shared<ServiceCosts>();
    auto constraint = std::make_shared<InsertionConstraint>();

    skipLines(input, 4);
    auto vehicle = readFleet(input, *fleet, *matrix);
    skipLines(input, 4);
    auto jobs = readJobs(input, *fleet, *matrix, vehicle);

    matrix->generate();

    (*constraint)
      .add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, matrix, activity))
      .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>());

    return std::make_shared<models::Problem>(
      models::Problem{fleet,
                      std::make_shared<models::problem::Jobs>(*matrix, *fleet, ranges::view::all(jobs)),
                      std::make_shared<std::vector<models::Lock>>(),
                      constraint,
                      std::make_shared<algorithms::objectives::penalize_unassigned_jobs<>>(),
                      activity,
                      matrix});
  }

private:
  /// Skips selected amount of lines from stream.
  void skipLines(std::istream& input, int count) const {
    ranges::for_each(ranges::view::ints(0, count),
                     [&input](auto) { input.ignore(std::numeric_limits<std::streamsize>::max(), input.widen('\n')); });
  }

  std::tuple<int, int> readFleet(std::istream& input,
                                 models::problem::Fleet& fleet,
                                 RoutingMatrix<Distance>& matrix) const {
    auto type = std::tuple<int, int>{};

    std::string line;
    std::getline(input, line);
    std::istringstream iss(line);
    iss >> std::get<0>(type) >> std::get<1>(type);

    fleet.add(models::problem::build_driver{}.dimens({{"id", std::string("driver")}}).costs({0, 0, 0, 0}).owned());
    return type;
  }

  std::vector<models::problem::Job> readJobs(std::istream& input,
                                             models::problem::Fleet& fleet,
                                             RoutingMatrix<Distance>& matrix,
                                             const std::tuple<int, int>& vehicle) const {
    constexpr models::common::Profile DefaultProfile = 0;
    /// Customer defined by: id, x, y, demand, start, end, service
    using CustomerData = std::tuple<int, int, int, int, int, int, int>;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;

    auto jobs = std::vector<models::problem::Job>{};

    auto customer = CustomerData{};
    auto last = -1;
    while (input) {
      input >> std::get<0>(customer) >> std::get<1>(customer) >> std::get<2>(customer) >>  //
        std::get<3>(customer) >> std::get<4>(customer) >> std::get<5>(customer) >> std::get<6>(customer);

      // skip last newlines
      if (!jobs.empty() && std::get<0>(customer) == last) break;
      last = std::get<0>(customer);
      int id = std::get<0>(customer);
      int location = matrix.location(std::get<1>(customer), std::get<2>(customer));

      if (id == 0) {
        ranges::for_each(ranges::view::ints(0, std::get<0>(vehicle)), [&](auto i) {
          fleet.add(
            models::problem::build_vehicle{}
              .profile(DefaultProfile)
              .costs({0, 1, 0, 0, 0})
              .dimens({{"id", std::string("v") + std::to_string(i + 1)}, {DimKeyCapacity, std::get<1>(vehicle)}})
              .details(
                {{0,
                  0,
                  {static_cast<Timestamp>(std::get<4>(customer)), static_cast<Timestamp>(std::get<5>(customer))}}})
              .owned());
        });
        continue;
      }

      jobs.push_back(as_job(
        build_service{}
          .dimens({{DimKeyDemand, Demand{{0, 0}, {std::get<3>(customer), 0}}},
                   {IdDimKey, std::string("c") + std::to_string(id)}})
          .details({{location,
                     static_cast<Duration>(std::get<6>(customer)),
                     {{static_cast<Timestamp>(std::get<4>(customer)), static_cast<Timestamp>(std::get<5>(customer))}}}})
          .shared()));
    }

    return std::move(jobs);
  }
};
}
