#pragma once

#include "algorithms/construction/constraints/ActorActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/Problem.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "streams/in/extensions/SolomonCosts.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <istream>
#include <map>
#include <range/v3/all.hpp>
#include <sstream>
#include <tuple>
#include <vector>

namespace vrp::streams::in {

/// Reads Li & Lim's PDPTW benchmark problems.
template<typename Distance = cartesian_distance>
struct read_li_lim_type final {
private:
  constexpr static auto IdDimKey = "id";
  inline static auto DimKeyCapacity = algorithms::construction::VehicleActivitySize<int>::DimKeyCapacity;
  inline static auto DimKeyDemand = algorithms::construction::VehicleActivitySize<int>::DimKeyDemand;

  using Capacity = algorithms::construction::VehicleActivitySize<int>::Capacity;
  using Demand = algorithms::construction::VehicleActivitySize<int>::Demand;

  /// Represents customer.
  struct Customer final {
    models::common::Location location;
    models::common::Duration duration;
    int size;
    models::common::TimeWindow tw;
  };

  /// Represents relation.
  struct Relation final {
    std::string pickup;
    std::string delivery;
  };

public:
  std::shared_ptr<models::Problem> operator()(std::istream& input) const {
    using namespace algorithms::construction;

    auto matrix = std::make_shared<RoutingMatrix<Distance>>();
    auto fleet = std::make_shared<models::problem::Fleet>();
    auto activity = std::make_shared<ServiceCosts>();
    auto constraint = std::make_shared<InsertionConstraint>();

    auto vehicle = readVehicleType(input, *fleet, *matrix);
    auto jobs = readJobs(input, *fleet, *matrix, vehicle);

    matrix->generate();

    (*constraint)
      .add<ActorActivityTiming>(std::make_shared<ActorActivityTiming>(fleet, matrix, activity))
      .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>());

    return std::make_shared<models::Problem>(
      models::Problem{fleet,
                      std::make_shared<models::problem::Jobs>(*matrix, *fleet, ranges::view::all(jobs)),
                      constraint,
                      std::make_shared<algorithms::objectives::penalize_unassigned_jobs<>>(),
                      activity,
                      matrix});
  }

private:
  std::tuple<int, int> readVehicleType(std::istream& input,
                                       models::problem::Fleet& fleet,
                                       RoutingMatrix<Distance>& matrix) const {
    auto type = std::tuple<int, int, int>{};

    std::string line;
    std::getline(input, line);
    std::istringstream iss(line);
    iss >> std::get<0>(type) >> std::get<1>(type) >> std::get<2>(type);

    fleet.add(models::problem::build_driver{}.dimens({{IdDimKey, std::string("driver")}}).costs({0, 0, 0, 0}).owned());
    return {std::get<0>(type), std::get<1>(type)};
  }

  std::vector<models::problem::Job> readJobs(std::istream& input,
                                             models::problem::Fleet& fleet,
                                             RoutingMatrix<Distance>& matrix,
                                             const std::tuple<int, int>& vehicle) const {
    /// Customer defined by: id, x, y, size, start, end, service, ?, relation
    using CustomerData = std::tuple<int, int, int, int, int, int, int, int, int>;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;
    using namespace ranges;

    // read problem
    auto customers = std::map<std::string, Customer>{};
    auto relations = std::vector<Relation>{};
    while (input) {
      auto cst = CustomerData{};
      input >> std::get<0>(cst) >> std::get<1>(cst) >> std::get<2>(cst) >> std::get<3>(cst) >> std::get<4>(cst) >>
        std::get<5>(cst) >> std::get<6>(cst) >> std::get<7>(cst) >> std::get<8>(cst);

      auto id = std::string("c") + std::to_string(std::get<0>(cst));
      auto location = matrix.location(std::get<1>(cst), std::get<2>(cst));
      int size = std::get<3>(cst);
      auto tw = TimeWindow{static_cast<Timestamp>(std::get<4>(cst)), static_cast<Timestamp>(std::get<5>(cst))};
      auto duration = static_cast<Duration>(std::get<6>(cst));
      auto relation = std::string("c") + std::to_string(std::get<8>(cst));

      // TODO find better way to stop
      if (tw.end == 0) break;

      if (id == "c0") {
        Capacity capacity = std::get<1>(vehicle);
        ranges::for_each(ranges::view::ints(0, std::get<0>(vehicle)), [&](auto i) {
          fleet.add(models::problem::build_vehicle{}
                      .profile("car")
                      .costs({0, 1, 0, 0, 0})
                      .dimens({{IdDimKey, std::string("v") + std::to_string(i + 1)}, {DimKeyCapacity, capacity}})
                      .details({{0, 0, {tw}}})
                      .owned());
        });
      } else if (size > 0) {
        relations.push_back(Relation{id, relation});
      }

      customers[id] = Customer{location, duration, std::abs(size), tw};
    }

    // create sequence jobs
    auto jobs = std::vector<models::problem::Job>{};
    ranges::for_each(view::zip(relations, view::iota(0)), [&](const auto& view) {
      auto [relation, index] = view;
      auto pickup = customers[relation.pickup];
      auto delivery = customers[relation.delivery];
      auto seqId = std::string("seq") + std::to_string(index);
      auto sequence =
        build_sequence{}
          .dimens({{IdDimKey, seqId}})
          .service(build_service{}
                     .dimens({{DimKeyDemand, Demand{{0, pickup.size}, {0, 0}}}, {IdDimKey, relation.pickup}})
                     .details({{pickup.location, pickup.duration, {pickup.tw}}})
                     .shared())
          .service(build_service{}
                     .dimens({{DimKeyDemand, Demand{{0, 0}, {0, delivery.size}}}, {IdDimKey, relation.delivery}})
                     .details({{delivery.location, delivery.duration, {delivery.tw}}})
                     .shared())
          .shared();
      jobs.push_back(as_job(sequence));
    });

    return std::move(jobs);
  }
};
}
