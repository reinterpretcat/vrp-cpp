#ifndef VRP_UTILS_SETTINGSFACTORY_HPP
#define VRP_UTILS_SETTINGSFACTORY_HPP

#include "algorithms/convolutions/Models.hpp"
#include "algorithms/genetic/Models.hpp"
#include "utils/Pool.hpp"

namespace vrp {
namespace test {

/// Creates convolution settings.
inline vrp::algorithms::convolutions::Settings createConvolutionSettings(float median,
                                                                         float convolution) {
  static vrp::utils::Pool pool;
  return {median, convolution, pool};
}

/// Creates genetic settings.
inline vrp::algorithms::genetic::Settings createGeneticSettings(int populationSize) {
  return {populationSize, createConvolutionSettings(0, 0)};
}

}  // namespace test
}  // namespace vrp

#endif  // VRP_UTILS_SETTINGSFACTORY_HPP
