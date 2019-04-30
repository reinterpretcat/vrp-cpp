#pragma once

#include "algorithms/construction/extensions/Factories.hpp"
#include "algorithms/construction/heuristics/CheapestInsertion.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Problem.hpp"
#include "models/solution/Registry.hpp"
#include "utils/extensions/Ranges.hpp"

#include <memory>
#include <numeric>

namespace vrp::algorithms::refinement {

/// TODO make constraint codes standard
constexpr static int JobLockConstraintCode = 4;

/// Creates refinement context including initial solution from problem.
template<typename Heuristic = construction::CheapestInsertion>
struct create_refinement_context final {
  RefinementContext operator()(const std::shared_ptr<const models::Problem>& problem) const {
    using namespace ranges;
    using namespace vrp::algorithms::construction;
    using namespace vrp::models;
    using Population = RefinementContext::Population;

    // TODO remove seed for production use
    auto random = std::make_shared<utils::Random>(0);
    auto registry = std::make_shared<models::solution::Registry>(*problem->fleet);

    // get various job types and construct initial routes
    auto unassignedJobs = std::map<problem::Job, int, problem::compare_jobs>{};
    auto lockedJobs = std::make_shared<JobSet>();
    auto reservedJobs = JobSet{};
    auto initRoutes = createInitialRoutes(*problem, *registry, unassignedJobs, lockedJobs, reservedJobs);
    auto requiredJobs = createRequiredJobs(*problem, unassignedJobs, lockedJobs, reservedJobs) | to_vector;

    // create initial solution represented by insertion context.
    auto iCtx = Heuristic{InsertionEvaluator{}}(
      build_insertion_context{}
        .progress(build_insertion_progress{}
                    .cost(models::common::NoCost)
                    .completeness(0)  // TODO recalculate it to include jobs from initial routes.
                    .total(static_cast<int>(problem->jobs->size()))
                    .owned())
        .problem(problem)
        .random(random)
        .solution(build_insertion_solution_context{}
                    .required(std::move(requiredJobs))
                    .routes(std::move(initRoutes))
                    .unassigned(std::move(unassignedJobs))
                    .registry(registry)
                    .shared())
        .owned());

    // create solution and calculate its cost
    auto sln = std::make_shared<models::Solution>(
      models::Solution{iCtx.solution->registry,
                       iCtx.solution->routes | view::transform([](const auto& rs) {
                         return static_cast<std::shared_ptr<const models::solution::Route>>(rs.route);
                       }) |
                         to_vector,
                       std::move(iCtx.solution->unassigned)});
    auto cost = problem->objective->operator()(*sln, *problem->activity, *problem->transport);

    return RefinementContext{
      problem, random, lockedJobs, std::make_shared<Population>(Population{models::EstimatedSolution{sln, cost}}), 0};
  }

private:
  using JobSet = std::set<models::problem::Job, models::problem::compare_jobs>;
  using InitRoutes = std::set<algorithms::construction::InsertionRouteContext,
                              algorithms::construction::compare_insertion_route_contexts>;

  /// Creates initial routes.
  InitRoutes createInitialRoutes(const models::Problem& problem,
                                 models::solution::Registry& registry,
                                 std::map<models::problem::Job, int, models::problem::compare_jobs>& unassignedJobs,
                                 std::shared_ptr<JobSet> lockedJobs,
                                 JobSet& reservedJobs) const {
    using namespace ranges;
    using namespace vrp::algorithms::construction;
    using namespace vrp::models;
    using namespace vrp::models::problem;
    using namespace vrp::models::solution;

    // TODO check that actor lock constraint is added when there is at least one jobs lock

    auto initRoutes = InitRoutes{};
    auto sequenceJobUsage = std::map<Job, std::size_t, compare_jobs>{};

    ranges::for_each(*problem.locks, [&](const auto& lock) {
      auto actor = utils::first<const Registry::SharedActor>(
        registry.available() | view::filter([&](const auto& actor) { return lock.condition(*actor); }));

      if (actor) {
        // create route context and add all jobs as they defined
        auto rs = create_insertion_route_context{}(actor.value());

        ranges::accumulate(
          lock.details, rs.route->tour.start()->detail.location, [&](const auto& acc, const auto& detail) {
            auto createActivity = [&](const auto& service) {
              assert(service->details.size() == 1);
              assert(service->details.front().times.size() == 1);

              const auto& d = service->details.front();
              return std::make_shared<Activity>(
                Activity{{d.location.value_or(acc), d.duration, d.times.front()}, {}, service});
            };

            // copy jobs to avoid them to be added as required
            if (detail.order != Lock::Order::Any)
              ranges::copy(detail.jobs, ranges::inserter(*lockedJobs, lockedJobs->begin()));
            else
              ranges::copy(detail.jobs, ranges::inserter(reservedJobs, reservedJobs.begin()));

            return ranges::accumulate(detail.jobs, 0, [&](const auto&, const auto& job) {
              auto activity = analyze_job<std::shared_ptr<Activity>>(
                job,
                [&](const std::shared_ptr<const Service>& srv) { return createActivity(srv); },
                [&](const std::shared_ptr<const problem::Sequence>& seq) {
                  assert(!seq->services.empty());
                  auto usage = sequenceJobUsage.find(job);
                  auto index = usage == sequenceJobUsage.end() ? std::size_t{0} : sequenceJobUsage[job];

                  sequenceJobUsage[job] = index + 1;

                  return createActivity(seq->services.at(index));
                });

              rs.route->tour.insert(activity);
              return activity->detail.location;
            });
          });

        // NOTE actor is marked as used in actor job lock.
        problem.constraint->accept(rs);
        initRoutes.insert(rs);

      } else {
        // add all jobs to unassigned
        ranges::for_each(lock.details, [&](const auto& detail) {
          ranges::for_each(detail.jobs,
                           [&](const auto& job) { unassignedJobs.insert(std::make_pair(job, JobLockConstraintCode)); });
        });
      }
    });

    // NOTE all services from sequence should be used in init route or none of them
    ranges::for_each(sequenceJobUsage, [&](const auto& pair) {
      analyze_job<void>(
        pair.first,
        [&](const std::shared_ptr<const Service>& srv) { assert(false); },
        [&](const std::shared_ptr<const problem::Sequence>& seq) { assert(seq->services.size() == pair.second); });
    });

    return std::move(initRoutes);
  }

  /// Creates required jobs.
  auto createRequiredJobs(const models::Problem& problem,
                          const std::map<models::problem::Job, int, models::problem::compare_jobs>& unassignedJobs,
                          const std::shared_ptr<JobSet>& lockedJobs,
                          const JobSet& reservedJobs) const {
    return problem.jobs->all() | ranges::view::filter([&](const auto& j) {
             return lockedJobs->find(j) == lockedJobs->end() &&  //
               reservedJobs.find(j) == reservedJobs.end() &&     //
               unassignedJobs.find(j) == unassignedJobs.end();
           });
  }
};
}
