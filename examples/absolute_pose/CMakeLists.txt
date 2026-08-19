include_directories(.)

add_subdirectory(problem_generator)
add_subdirectory(affine2sift)

add_executable(test_random_problems test_random_problems.cpp)
target_link_libraries(test_random_problems eccv2026 problem_generator affine2sift PoseLib::PoseLib)

