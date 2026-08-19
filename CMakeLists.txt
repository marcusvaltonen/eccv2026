cmake_minimum_required(VERSION 3.2)
project(ECCV2026)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


SET (CMAKE_BUILD_TYPE "RELEASE")

if (CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
endif(CMAKE_COMPILER_IS_GNUCXX)

find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIR} )

include_directories(
    src/helpers
    src/include
)

set(SRCS
    src/helpers/gj.cpp
    src/solvers/solver_p2p_ori.cpp
    src/solvers/solver_p35pf.cpp
    src/solvers/solver_up1pf_ac.cpp
    src/solvers/solver_acp1p_cayley.cpp
    src/solvers/solver_up1p_sift.cpp
    src/solvers/solver_up2pf_ori.cpp
    src/solvers/solver_up2p.cpp
    src/solvers/solver_up3pf.cpp
    src/solvers/solver_up3pfr.cpp
    src/solvers/solver_p4pf.cpp
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(eccv2026 SHARED ${SRCS})
target_link_libraries(eccv2026 PRIVATE Eigen3::Eigen)

find_package( PoseLib REQUIRED )
target_link_libraries(eccv2026 PRIVATE PoseLib::PoseLib)
include_directories(${CMAKE_SOURCE_DIR})

# Examples
add_subdirectory(examples)

# Tests
option(WITH_TEST "Build test suite." OFF)
if (WITH_TEST)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        release-1.11.0
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest INTERFACE IMPORTED)
    target_link_libraries(GTest::GTest INTERFACE gtest_main)

    add_executable(eccv2026_test tests/eccv2026_test.cpp ${SRCS})
    target_link_libraries(eccv2026_test PRIVATE GTest::GTest)
    target_link_libraries(eccv2026_test PRIVATE Eigen3::Eigen)
    target_link_libraries(eccv2026_test PRIVATE PoseLib::PoseLib)
    add_test(eccv2026_test eccv2026_test)
endif()
