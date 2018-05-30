#include "algorithms/genetic/Crossovers.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "test_utils/PopulationFactory.hpp"
#include "test_utils/SolomonBuilder.hpp"

#include <algorithms/convolutions/SlicedConvolutions.hpp>
#include <catch/catch.hpp>

using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::genetic;
using namespace vrp::streams;
using namespace vrp::utils;
using namespace vrp::test;

namespace {
std::pair<vrp::models::Problem, vrp::models::Tasks> getPopulation(int populationSize) {
  auto stream = SolomonBuilder()
                  .setTitle("Exceeded capacity and two vehicles")
                  .setVehicle(25, 200)
                  .addCustomer({0, 40, 50, 0, 0, 1236, 0})
                  .addCustomer({1, 45, 68, 10, 912, 967, 90})
                  .addCustomer({2, 45, 70, 30, 825, 870, 90})
                  .addCustomer({3, 42, 66, 10, 65, 146, 90})
                  .addCustomer({4, 42, 68, 10, 727, 782, 90})
                  .addCustomer({5, 42, 65, 10, 15, 67, 90})
                  .addCustomer({6, 40, 69, 20, 621, 702, 90})
                  .addCustomer({7, 40, 66, 20, 170, 225, 90})
                  .addCustomer({8, 38, 68, 20, 255, 324, 90})
                  .addCustomer({9, 38, 70, 10, 534, 605, 90})
                  .addCustomer({10, 35, 66, 10, 357, 410, 90})
                  .addCustomer({11, 35, 69, 10, 448, 505, 90})
                  .addCustomer({12, 25, 85, 20, 652, 721, 90})
                  .addCustomer({13, 22, 75, 30, 30, 92, 90})
                  .addCustomer({14, 22, 85, 10, 567, 620, 90})
                  .addCustomer({15, 20, 80, 40, 384, 429, 90})
                  .addCustomer({16, 20, 85, 40, 475, 528, 90})
                  .addCustomer({17, 18, 75, 20, 99, 148, 90})
                  .addCustomer({18, 15, 75, 20, 179, 254, 90})
                  .addCustomer({19, 15, 80, 10, 278, 345, 90})
                  .addCustomer({20, 30, 50, 10, 10, 73, 90})
                  .addCustomer({21, 30, 52, 20, 914, 965, 90})
                  .addCustomer({22, 28, 52, 20, 812, 883, 90})
                  .addCustomer({23, 28, 55, 10, 732, 777, 90})
                  .addCustomer({24, 25, 50, 10, 65, 144, 90})
                  .addCustomer({25, 25, 52, 40, 169, 224, 90})
                  .build();
  return createPopulation<>(stream, populationSize);
};
}  // namespace

SCENARIO("Can create offsprings", "[genetic][crossover][acdc]") {
  //  int populationSize = 4;
  //  auto population = getPopulation(populationSize);
  //
  //  auto result = adjusted_cost_difference{}.operator()(
  //    population.first, population.second,
  //    createGeneticSettings(populationSize, createConvolutionSettings(0.5, 0.05)), {{0, 1}, {2,
  //    3}});

  create_sliced_convolutions{}.operator()(/*problem, tasks, settings.convolution, pairs*/);


  // TODO
  // MatrixTextWriter().write(std::cout, population.first, population.second);
}
