# Description

A prototype of VRP solver with custom algorithm


# Problem

The following VRP variation is addressed:
* homogeneous fleet
* single depot with time window
* vehicle time constraint
* customers with demand and time window
* no breaks, no refueling, no pickups, etc.


# Algorithm

The main idea is to use genetic algorithm variation with original crossover operator.


# Status

In development. Not ready for use.


# Remarks

* tested with cuda compilation tools, release 9.1, V9.1.85 and thrust 1.8.3
* depot time window is not less than vehicles operating time limit.
* no unfeasable customers while building initial solution.


# Benchmarks

* see https://www.sintef.no/projectweb/top/vrptw/
* run solver runner with in/out file arguments, e.g.:
        ../../resources/data/solomon/benchmarks/RC1_10_1.txt result.json
