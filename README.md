# Description

A prototype of VRP solver


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
    * work on vNext (rich) format
        * use real RFC3339 times
        * use real locations instead of ids
        * add breaks
        * use polygon to specify driver location?

# Check
    * Check that real dates can be handled
    * Check break:
       ruin handles conditional jobs correctly
       CreateRefinementContext adds all jobs as required

## TechDebt
    * init routes: move rule creation logic from actor_job_lock to create_refinement_context
    * why routes are set in InsertionContext?
    * rename get_job_ids_from_jobs and etc.
    * update dependencies
    * build with different compilers/versions

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

# Docker
    docker build -t solverex .
    docker run -it -v $(pwd):/app --rm solverex