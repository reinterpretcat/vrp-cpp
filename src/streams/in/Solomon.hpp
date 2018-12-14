#pragma once

#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/construction/constraints/VehicleActivityTiming.hpp"
#include "models/Problem.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Factories.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <istream>
#include <range/v3/all.hpp>
#include <sstream>
#include <tuple>

namespace vrp::streams::in {

/// Calculates cartesian distance between two points on plane in 2D.
template<unsigned int Scale = 1>
struct cartesian_distance final {
  models::common::Distance operator()(const std::pair<int, int>& left, const std::pair<int, int>& right) {
    auto x = left.first - right.first;
    auto y = left.second - right.second;
    return static_cast<models::common::Distance>(std::round(std::sqrt(x * x + y * y) * Scale));
  }
};

/// Reads problem represented by classical solomon definition from stream.
template<typename Distance = cartesian_distance<1>>
struct read_solomon_type final {
  constexpr static auto SizeDimKey = "size";

  struct ServiceCosts : models::costs::ActivityCosts {
    models::common::Cost cost(const models::solution::Actor& actor,
                              const models::solution::Activity& activity,
                              const models::common::Timestamp arrival) const override {
      return 0;
    }
  };

  struct RoutingMatrix : models::costs::TransportCosts {
    friend read_solomon_type;

    models::common::Duration duration(const std::string& profile,
                                      const models::common::Location& from,
                                      const models::common::Location& to,
                                      const models::common::Timestamp& departure) const override {
      return distance(profile, from, to, departure);
    }

    models::common::Distance distance(const std::string&,
                                      const models::common::Location& from,
                                      const models::common::Location& to,
                                      const models::common::Timestamp&) const override {
      return matrix_[from * locations_.size() + to];
    }


    auto matrix() const { return ranges::view::all(matrix_); }

  private:
    models::common::Location location(int x, int y) {
      // TODO use more performant data structure to have O(1)
      auto location = std::find_if(
        locations_.begin(), locations_.end(), [&](const auto& l) { return l.first == x && l.second == y; });

      if (location != locations_.end())
        return static_cast<models::common::Location>(std::distance(locations_.begin(), location));

      locations_.push_back(std::pair(x, y));
      return locations_.size() - 1;
    }

    void generate() {
      matrix_.reserve(locations_.size() * locations_.size());

      auto distance = Distance{};
      for (size_t i = 0; i < locations_.size(); ++i)
        for (size_t j = 0; j < locations_.size(); ++j) {
          matrix_.push_back(i != j ? distance(locations_[i], locations_[j]) : static_cast<models::common::Distance>(0));
        }
    }

    std::vector<models::common::Distance> matrix_;
    std::vector<std::pair<int, int>> locations_;
  };

  models::Problem operator()(std::istream& input) const {
    using namespace algorithms::construction;

    auto matrix = std::make_shared<RoutingMatrix>();
    auto fleet = std::make_shared<models::problem::Fleet>();
    auto activity = std::make_shared<ServiceCosts>();
    auto constraint = std::make_shared<InsertionConstraint>();
    (*constraint)
      .addHard<VehicleActivityTiming>(std::make_shared<VehicleActivityTiming>(fleet, matrix, activity))
      .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>());

    skipLines(input, 4);
    auto vehicle = readFleet(input, *fleet, *matrix);
    skipLines(input, 4);
    auto jobs = readJobs(input, *fleet, *matrix, vehicle);

    matrix->generate();

    return {fleet,
            std::make_shared<models::problem::Jobs>(*matrix, ranges::view::all(jobs), ranges::view::single("car")),
            constraint,
            activity,
            matrix};
  }

private:
  /// Skips selected amount of lines from stream.
  void skipLines(std::istream& input, int count) const {
    ranges::for_each(ranges::view::ints(0, count),
                     [&input](auto) { input.ignore(std::numeric_limits<std::streamsize>::max(), input.widen('\n')); });
  }

  std::tuple<int, int> readFleet(std::istream& input, models::problem::Fleet& fleet, RoutingMatrix& matrix) const {
    auto type = std::tuple<int, int>{};

    std::string line;
    std::getline(input, line);
    std::istringstream iss(line);
    iss >> std::get<0>(type) >> std::get<1>(type);

    fleet.add(models::problem::build_driver{}.id("driver").costs({0, 0, 0, 0}).owned());
    return type;
  }

  std::set<models::problem::Job, models::problem::compare_jobs> readJobs(std::istream& input,
                                                                         models::problem::Fleet& fleet,
                                                                         RoutingMatrix& matrix,
                                                                         const std::tuple<int, int>& vehicle) const {
    /// Customer defined by: id, x, y, demand, start, end, service
    using CustomerData = std::tuple<int, int, int, int, int, int, int>;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;

    auto jobs = std::set<Job, compare_jobs>{};

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
          fleet.add(models::problem::build_vehicle{}
                      .id(std::string("v") + std::to_string(i + 1))
                      .costs({0, 1, 0, 0, 0})
                      .dimens({{SizeDimKey, std::get<1>(vehicle)}})
                      .details({{0,
                                 0,
                                 {static_cast<Timestamp>(std::get<4>(customer)),
                                  static_cast<Timestamp>(std::get<5>(customer))}}})
                      .owned());
        });
        continue;
      }

      jobs.insert(as_job(
        build_service{}
          .id(std::string("c") + std::to_string(id))
          .dimens({{SizeDimKey, -std::get<3>(customer)}})
          .details({{location,
                     static_cast<Duration>(std::get<6>(customer)),
                     {{static_cast<Timestamp>(std::get<4>(customer)), static_cast<Timestamp>(std::get<5>(customer))}}}})
          .shared()));
    }

    return std::move(jobs);
  }
};
}
