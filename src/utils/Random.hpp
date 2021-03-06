#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <range/v3/all.hpp>
#include <type_traits>

namespace vrp::utils {

/// Provides the way to use randomized values in generic way.
class Random final {
public:
  /// Specifies int distribution which generates values on closed [min, max] interval.
  using IntDistribution = std::function<int(int min, int max)>;

  /// Specifies real distribution which generates values on closed [min, max] interval.
  using RealDistribution = std::function<double(double min, double max)>;

  /// Creates random with default int and real distributions.
  Random() :
    generator_(std::random_device()()),
    intDist_(std::bind(&Random::intDist, this, std::placeholders::_1, std::placeholders::_2)),
    realDist_(std::bind(&Random::realDist, this, std::placeholders::_1, std::placeholders::_2)) {}

  /// Creates random with custom integral and real distributions.
  Random(IntDistribution intDist, RealDistribution realDist) :
    generator_(std::random_device()()),
    intDist_(std::move(intDist)),
    realDist_(std::move(realDist)) {}

  explicit Random(std::mt19937::result_type seed) : Random() { generator_.seed(seed); }

  /// Produces integral random value, uniformly distributed on the closed interval [min, max]
  template<class T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  T uniform(T min, T max) {
    return intDist_(min, max);
  }

  /// Produces real random value, uniformly distributed on the closed interval [min, max]
  template<class T, typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
  T uniform(T min, T max) {
    return realDist_(min, max);
  }

  /// Flips a coin and returns true if it is "heads", false otherwise.
  bool isHeadsNotTails() { return uniform<int>(1, 2) == 1; }

  template<typename Iterator>
  void shuffle(Iterator begin, Iterator end) {
    std::shuffle(begin, end, generator_);
  }

  /// Returns an index from collected with probability weight.
  /// Use exponential distribution where the weights are the rate of the distribution (lambda)
  /// and select the smallest sampled value.
  int weighted(ranges::any_view<int> weights) {
    using namespace ranges;

    return ranges::min(view::zip(weights, view::iota(0)) | view::transform([&](auto pair) {
                         return std::pair(-std::log(uniform<double>(0, 1)) / pair.first, pair.second);
                       }),
                       [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; })
      .second;
  }

private:
  int intDist(int min, int max) { return std::uniform_int_distribution<int>(min, max)(generator_); }

  double realDist(double min, double max) { return std::uniform_real_distribution<double>(min, max)(generator_); }

  std::mt19937 generator_;
  IntDistribution intDist_;
  RealDistribution realDist_;
};
}