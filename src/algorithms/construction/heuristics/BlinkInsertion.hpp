#pragma once

#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/InsertionHeuristic.hpp"
#include "algorithms/construction/InsertionResult.hpp"
#include "algorithms/construction/extensions/Sorters.hpp"

#include <array>
#include <functional>
#include <mutex>

namespace vrp::algorithms::construction {

/// Selects best result with blinks where ratio is defined by nominator / denominator.
template<int Nominator, int Denominator>
struct select_insertion_with_blinks final {
  constexpr static double ratio = double(Nominator) / Denominator;

  explicit select_insertion_with_blinks(const InsertionContext& ctx) : ctx_(ctx), lock_() {}

  InsertionResult operator()(const InsertionResult& left, const InsertionResult& right) const {
    // NOTE cannot blink having invalid left result
    return left.index() == 0 && isBlink() ? left : get_best_result(left, right);
  }

private:
  /// Checks whether new job insertion result should be ignored.
  bool isBlink() const {
    std::lock_guard<std::mutex> lock(lock_);
    return ctx_.random->uniform<double>(0, 1) < ratio;
  }
  const InsertionContext& ctx_;
  mutable std::mutex lock_;
};

/// Selects jobs range based on SISR rules.
template<typename Size>
struct select_insertion_range_blinks final {
  using Sorter = std::function<void(InsertionContext&)>;

  random_jobs_sorter random = {};
  sized_jobs_sorter<Size> sizedDesc = {true};
  sized_jobs_sorter<Size> sizedAsc = {false};
  ranked_jobs_sorter rankedDesc = {true};
  ranked_jobs_sorter rankedAsc = {false};

  /// Keeps sorters within their weights.
  std::array<std::pair<Sorter, int>, 5> sorters = {std::pair(Sorter(std::ref(random)), 10),
                                                   std::pair(Sorter(std::ref(sizedDesc)), 10),
                                                   std::pair(Sorter(std::ref(sizedAsc)), 1),
                                                   std::pair(Sorter(std::ref(rankedAsc)), 5),
                                                   std::pair(Sorter(std::ref(rankedDesc)), 1)};

  auto operator()(InsertionContext& ctx) const {
    using namespace ranges;

    constexpr int minSize = 8;
    constexpr int maxSize = 16;

    sorters.at(ctx.random->weighted(sorters | view::transform([](const auto& p) { return p.second; }))).first(ctx);

    auto sampleSize = std::min(static_cast<int>(ctx.jobs.size()), ctx.random->uniform<int>(minSize, maxSize));

    return std::pair(ctx.jobs.begin(), ctx.jobs.begin() + sampleSize);
  }
};

/// Specifies insertion with blinks heuristic.
/// NOTE insertion heuristics processes all jobs simultaneously, so
/// sorting part by different customer property (e.g. demand, far, close) from the
/// original paper is omitted.
template<int Nominator = 1, int Denominator = 100, typename Size = int>
using BlinkInsertion = InsertionHeuristic<InsertionEvaluator,
                                          select_insertion_range_blinks<Size>,
                                          select_insertion_with_blinks<Nominator, Denominator>>;
}