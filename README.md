# Description

A prototype of VRP solver
https://github.com/justusc/FindTBB


# Install

    # build tbb from https://github.com/philipp-classen/tbb-static-linking-tutorial
    make extra_inc=big_iron.inc

    sudo apt install libtbb-dev
    conan install parallelstl/20181004@conan/stable


# Objective function
    * minimize load
        * minimize the number of vehicles
        * minimize the total completion time
        * minimize the total interval completion time
            (reduce the difference between the total completion time of the
             longest tour and the total completion time of the shortest tour)
            minF = w0*NV + w1*TCT + w2*RCT
    * minimize cost


# TODO

## Next
    * generate here solution
    * initial routes
        * any
        * sequence
        * strict
    * Vehicle breaks
        * break with location [done]
        * break without location !!!
        * write helper method to generate test json [done]


# Check
    * Check that real dates can be handled
    * Check break:
       ruin handles conditional jobs correctly
       CreateRefinementContext adds all jobs as required

## TechDebt
    * clean headers from iostream
    * HereJsonTest: replace json text with json builder
    * merge two accept method from constraint:
        accept(SolutionContext, RouteContext, Job)? [No, can't do that]
    * why routes are set in InsertionContext?
    * rename get_job_ids_from_jobs and etc.
    * use snake case for ruin/recreate methods
    * use release tbb for release

## Various
    * Ruin jobs neighbours when they are not assigned?
    * Ruin jobs without location?

    * Fleet minimization
    * R&R with sequence
    * simplify registry
    * check movement syntax within builders to avoid copying
    * remove string selects job from activity (does not work for sequence)
    * improve jobs distances logic?

    * make expensive to allocate new actor during insertion by checking its tour?

    * sequence picks best time window only once ignoring other services

    * optimize tour activity insertion
    * optimize size constraint for empty tours?
    * hierarchical objective


# Potential use cases
    * Job with mixed demand
    * Multiday planning

# Ideas
    * compile to web assembly - run in browser