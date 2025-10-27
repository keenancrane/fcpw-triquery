cmake_minimum_required(VERSION 3.15...3.27)
project(fcpw-tests)

################################################################################
# build tests
list(APPEND FCPW_TESTS_INCLUDES ${FCPW_ARGS_INCLUDES})

# Try to find system TBB first, fall back to bundled version
find_package(TBB QUIET)
if(NOT TBB_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)
    if(FCPW_BUILD_TESTS)
        set(TBB_TEST          OFF CACHE BOOL " " FORCE)
        set(TBB_INSTALL       OFF CACHE BOOL " " FORCE)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)
        list(APPEND FCPW_TESTS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)
    endif()
endif()

# Only build tests that require polyscope if it's available
if(TARGET polyscope)
    add_executable(aggregate_tests aggregate_tests.cpp)
    target_link_libraries(aggregate_tests fcpw polyscope TBB::tbb)
    target_include_directories(aggregate_tests PRIVATE ${FCPW_TESTS_INCLUDES})

    add_executable(csg_tests csg_tests.cpp)
    target_link_libraries(csg_tests fcpw polyscope TBB::tbb)
    target_include_directories(csg_tests PRIVATE ${FCPW_TESTS_INCLUDES})

    if(FCPW_ENABLE_GPU_SUPPORT)
        add_executable(gpu_tests gpu_tests.cpp)
        target_link_libraries(gpu_tests fcpw polyscope TBB::tbb)
        target_include_directories(gpu_tests PRIVATE ${FCPW_TESTS_INCLUDES})
    endif()
endif()

# Triangle query tests don't need polyscope
if(TBB_FOUND OR TARGET TBB::tbb)
    add_executable(triangle_query_tests triangle_query_tests.cpp)
    target_link_libraries(triangle_query_tests fcpw TBB::tbb)
    target_include_directories(triangle_query_tests PRIVATE ${FCPW_TESTS_INCLUDES})
endif()
