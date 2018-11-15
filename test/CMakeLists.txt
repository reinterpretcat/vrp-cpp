project(SolverTest LANGUAGES CXX)

set(LIBRARY_NAME Solver)
set(LIBRARY_TEST SolverTest)

include_directories(${MAIN_SOURCE}
        ${EXTERNAL_SOURCE}/catch/src
        ${EXTERNAL_SOURCE}/range-v3/include
        ${TEST_SOURCE}
        ${CMAKE_CURRENT_BINARY_DIR})

set(SOURCE_FILES
        models/solution/TourTest.cc

        models/algorithms/construction/insertion/InsertionConstraintTest.cc
        models/algorithms/construction/insertion/evaluators/ServiceInsertionEvaluatorTest.cc
        models/algorithms/construction/insertion/evaluators/JobInsertionEvaluatorTest.hpp

        test_utils/algorithms/construction/Insertions.hpp
        test_utils/fakes/TestTransportCosts.hpp
        test_utils/fakes/TestActivityCosts.hpp
        test_utils/models/Factories.hpp
        test_utils/models/Matchers.hpp
        )

add_executable(${LIBRARY_TEST}
        main.cpp
        ${SOURCE_FILES}
        )


set_target_properties(${LIBRARY_TEST} PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION ON
        )

target_link_libraries(${LIBRARY_TEST} PRIVATE
        ${LIBRARY_NAME}
        )
