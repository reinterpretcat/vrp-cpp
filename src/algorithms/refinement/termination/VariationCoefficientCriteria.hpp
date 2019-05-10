#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/common/Cost.hpp"

#include <optional>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::algorithms::refinement {

/// Uses coefficient of variation as termination criteria.
struct VariationCoefficientCriteria final {
  explicit VariationCoefficientCriteria(std::size_t capacity, double threshold) :
    capacity_(capacity),
    capacityMinusOne_(capacity - 1),
    threshold_(threshold),
    last_(models::common::NoCost),
    costs_() {
    assert(capacity > 1);
    costs_.resize(capacity);
  }

  /// Returns true if algorithm should be terminated.
  bool operator()(const RefinementContext& ctx, const models::EstimatedSolution& solution, bool accepted) {
    // TODO do we need to consider penalties?
    if (accepted) last_ = solution.second.actual;

    // NOTE we start counting generations from 1
    costs_[(ctx.generation - 1) % capacity_] = last_;

    return ctx.generation >= capacity_ ? checkThreshold() : false;
  }

private:
  bool checkThreshold() const {
    auto sum = ranges::accumulate(costs_, .0, ranges::plus{});

    auto mean = sum / static_cast<double>(capacity_);

    auto variance = calculateVariance(mean);

    auto sdev = std::sqrt(variance);

    auto cv = sdev / mean;

    return cv < threshold_;
  }

  double calculateVariance(double mean) const {
    auto [first, second] = ranges::accumulate(costs_, std::pair<double, double>(.0, .0), [mean](auto acc, auto v) {
      auto dev = (v - mean);
      return std::make_pair(acc.first + dev * dev, acc.second + dev);
    });

    return (first - (second * second / capacity_)) / capacityMinusOne_;
  }

  std::size_t capacity_;
  std::size_t capacityMinusOne_;
  double threshold_;
  models::common::Cost last_;
  std::vector<models::common::Cost> costs_;
};
}