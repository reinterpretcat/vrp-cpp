#ifndef VRP_UTILS_SETTINGSFACTORY_HPP
#define VRP_UTILS_SETTINGSFACTORY_HPP

#include "algorithms/convolutions/Models.hpp"
#include "algorithms/genetic/Models.hpp"

namespace vrp {
namespace test {

/// Creates convolution settings.
inline vrp::algorithms::convolutions::Settings createConvolutionSettings(float median,
                                                                         float convolution) {
  return {median, convolution};
}

/// Creates genetic settings.
inline vrp::algorithms::genetic::Settings createGeneticSettings(int populationSize) {
  return {populationSize, createConvolutionSettings(0, 0)};
}

/// Creates genetic settings with user-defined convolution settings.
inline vrp::algorithms::genetic::Settings createGeneticSettings(
  int populationSize,
  const vrp::algorithms::convolutions::Settings& settings) {
  return {populationSize, settings};
}

}  // namespace test
}  // namespace vrp

#endif  // VRP_UTILS_SETTINGSFACTORY_HPP
